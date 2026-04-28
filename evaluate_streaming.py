"""
Full-corpus streaming evaluation of quantization schemes on BEIR embeddings.

Unlike evaluate_quantization.py (which loads everything into RAM), this script
processes the corpus in batches and accumulates top-k retrieval scores without
ever loading the full corpus at once.

Memory footprint (with --c-batch 30000, --q-batch 512):
  Query vectors:   ~756 MB (all FEVER queries, 123k × 1536 × f32)
  Running top-k:   ~150 MB (123k queries × 100 results × 12 bytes)
  One corpus batch: ~550 MB (raw + angles + quantized: 3 × 30k × 1536 × f32)
  Score matrix:      ~60 MB (512 queries × 30k docs × f32)
  Total peak:       ~1.6 GB  ← fits on t3.medium (4 GB)

Resumability:
  Pass 1 (stats):  saves stats_partial.npz after each batch; promotes to
                   stats.npz when complete.
  Pass 2 (score):  chunk-major loop; saves a single progress/scoring_state.npz
                   (top_scores + top_gidx for *all* schemes stacked) every
                   --checkpoint-every chunks. Restart the script to resume.

Usage:
    python evaluate_streaming.py \\
        --embed-dir ./embeddings/fever \\
        --data-dir  ~/bier-data/fever \\
        --embed-tag "" \\
        --output-dir ./eval_results_full
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# BLAS thread control (must run before heavy numpy work)
# ---------------------------------------------------------------------------

def configure_blas(n_threads: int | None = None) -> None:
    """Force the underlying BLAS lib (OpenBLAS / MKL / Accelerate / BLIS) to
    use n_threads. Default = os.cpu_count(). Reports detected libraries.
    Requires `pip install threadpoolctl`."""
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        print("  [blas] threadpoolctl not installed; BLAS may be single-threaded. "
              "Run: pip install threadpoolctl", flush=True)
        return
    target = n_threads or os.cpu_count() or 1
    info_before = threadpool_info()
    threadpool_limits(limits=target)
    info_after = threadpool_info()
    for b, a in zip(info_before, info_after):
        print(f"  [blas] {a.get('user_api','?'):>8} via {a.get('prefix','?'):<10} "
              f"{b['num_threads']} -> {a['num_threads']} threads", flush=True)


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def free_mem() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chunk discovery + loading (one chunk at a time, no concatenation)
# ---------------------------------------------------------------------------

def discover_chunks(split_dir: Path, tag: str) -> list[Path]:
    if tag:
        return sorted(split_dir.glob(f"chunk_*.{tag}.npy"))
    return sorted(p for p in split_dir.glob("chunk_*.npy")
                  if p.name.count(".") == 1)


def load_chunk(chunk_path: Path) -> tuple[np.ndarray, list[str]]:
    arr   = np.load(chunk_path).astype(np.float32, copy=False)
    stem  = chunk_path.name[:-len(".npy")]
    ids_p = chunk_path.parent / (stem + ".ids.txt")
    with ids_p.open("r", encoding="utf-8") as f:
        ids = [ln.rstrip("\n") for ln in f]
    return arr, ids


def load_queries(split_dir: Path, tag: str) -> tuple[np.ndarray, list[str]]:
    chunks = discover_chunks(split_dir, tag)
    vecs, ids = [], []
    for p in chunks:
        arr, chunk_ids = load_chunk(p)
        vecs.append(arr)
        ids.extend(chunk_ids)
    return np.concatenate(vecs, axis=0), ids


# ---------------------------------------------------------------------------
# Qrels
# ---------------------------------------------------------------------------

def load_qrels(data_dir: Path) -> dict[str, dict[str, int]]:
    for name in ("test.tsv", "dev.tsv", "train.tsv"):
        p = data_dir / "qrels" / name
        if p.exists():
            print(f"  qrels: {p}")
            break
    else:
        raise FileNotFoundError(f"No qrels in {data_dir / 'qrels'}")
    qrels: dict[str, dict[str, int]] = {}
    with p.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qid, cid, s = row["query-id"], row["corpus-id"], int(row["score"])
            if s > 0:
                qrels.setdefault(qid, {})[cid] = s
    return qrels


# ---------------------------------------------------------------------------
# Coordinate conversion (float32 throughout)
# ---------------------------------------------------------------------------

def to_angles(x: np.ndarray) -> np.ndarray:
    """Cartesian -> (n-1) hyperspherical angles. Fully vectorised."""
    x    = x.astype(np.float32, copy=False)
    N, n = x.shape
    sq   = x * x
    tail = np.sqrt(np.clip(np.cumsum(sq[:, ::-1], axis=1)[:, ::-1],
                           0.0, None)).astype(np.float32, copy=False)
    del sq
    out = np.empty((N, n - 1), dtype=np.float32)
    if n > 2:
        out[:, : n - 2] = np.arccos(np.clip(
            x[:, : n - 2] / np.clip(tail[:, : n - 2], 1e-7, None),
            -1.0, 1.0))
    last          = np.arctan2(x[:, n - 1], x[:, n - 2])
    out[:, n - 2] = np.where(last < 0, last + np.float32(2 * math.pi), last)
    del tail, last
    return out


def from_angles(a: np.ndarray) -> np.ndarray:
    """Inverse of to_angles. Vectorised via cumprod-of-sin."""
    a = a.astype(np.float32, copy=False)
    N, m = a.shape
    cosA = np.cos(a, dtype=np.float32)
    sinA = np.sin(a, dtype=np.float32)
    cum_sin = np.empty((N, m + 1), dtype=np.float32)
    cum_sin[:, 0]  = 1.0
    cum_sin[:, 1:] = np.cumprod(sinA, axis=1)
    del sinA
    x = np.empty((N, m + 1), dtype=np.float32)
    x[:, :m] = cosA * cum_sin[:, :m]
    x[:,  m] = cum_sin[:, m]
    del cosA, cum_sin
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-7, None)


def _uquant(angles: np.ndarray, bits: int, hi: np.ndarray) -> np.ndarray:
    levels = 1 << bits
    a = np.clip(angles, 0.0, hi - 1e-6)
    q = np.clip(np.floor(a / hi * levels).astype(np.int32), 0, levels - 1)
    return (q.astype(np.float32) + 0.5) * (hi / levels)


# ---------------------------------------------------------------------------
# Pass 1: streaming stats collection
# ---------------------------------------------------------------------------

def collect_stats(corpus_chunks: list[Path], stats_path: Path,
                  partial_path: Path, dim: int) -> dict:
    """
    Compute clip_means (mean of each angle over full corpus) and
    Jacobian sensitivity (mean log-sin-product chain) from all corpus chunks.

    Fully resumable: partial sums are saved after each chunk and reloaded
    on restart. When all chunks are processed, saves stats_path and returns.
    """
    n_angles = dim - 1

    if stats_path.exists():
        print(f"  [stats] loading cached stats from {stats_path}")
        d = np.load(stats_path)
        return {"clip_means": d["clip_means"], "sens": d["sens"],
                "jac_rank": d["jac_rank"], "hi": d["hi"]}

    # Load partial sums if resuming
    if partial_path.exists():
        d = np.load(partial_path)
        sum_angles     = d["sum_angles"]
        sum_log_sin2   = d["sum_log_sin2"]
        count          = int(d["count"])
        chunks_done    = int(d["chunks_done"])
        print(f"  [stats] resuming from chunk {chunks_done} "
              f"({count:,} docs processed)")
    else:
        sum_angles     = np.zeros(n_angles, dtype=np.float64)
        sum_log_sin2   = np.zeros(n_angles, dtype=np.float64)
        count          = 0
        chunks_done    = 0

    total = len(corpus_chunks)
    for ci, chunk_path in enumerate(corpus_chunks):
        if ci < chunks_done:
            continue
        c_batch, _ = load_chunk(chunk_path)
        angles = to_angles(c_batch)
        del c_batch

        sum_angles   += angles.sum(axis=0).astype(np.float64)
        lsin2 = 2.0 * np.log(np.clip(np.abs(np.sin(angles)), 1e-7, None))
        # Jacobian: cumulative log-sin-squared from the left
        cum = np.zeros_like(lsin2)
        cum[:, 1:] = np.cumsum(lsin2[:, :-1], axis=1)
        sum_log_sin2 += cum.mean(axis=0).astype(np.float64)
        count        += angles.shape[0]
        del angles, lsin2, cum
        free_mem()

        chunks_done = ci + 1
        np.savez(partial_path,
                 sum_angles=sum_angles, sum_log_sin2=sum_log_sin2,
                 count=count, chunks_done=chunks_done)
        print(f"  [stats] chunk {chunks_done}/{total}  ({count:,} docs)", end="\r")

    print()
    clip_means = (sum_angles / count).astype(np.float32)

    mean_log    = (sum_log_sin2 / (chunks_done)).astype(np.float64)
    sens        = np.exp(mean_log - mean_log.max()).astype(np.float32)
    jac_rank    = np.argsort(-sens).astype(np.int32)

    hi           = np.full(n_angles, math.pi, dtype=np.float32)
    hi[-1]       = 2 * math.pi

    np.savez(stats_path, clip_means=clip_means, sens=sens,
             jac_rank=jac_rank, hi=hi)
    partial_path.unlink(missing_ok=True)
    print(f"  [stats] complete. {count:,} docs.  saved → {stats_path}")
    return {"clip_means": clip_means, "sens": sens,
            "jac_rank": jac_rank, "hi": hi}


# ---------------------------------------------------------------------------
# Quantization scheme definitions
# ---------------------------------------------------------------------------

def make_schemes(stats: dict, dim: int) -> list[tuple[str, object, int]]:
    """
    Returns list of (name, fn, bits_per_vec) where:
        fn(c_batch, c_angles) -> quantized c_batch (float32, unit-normed)
    c_angles may be None for Cartesian-only schemes.
    """
    clip_means = stats["clip_means"]
    jac_rank   = stats["jac_rank"]
    hi         = stats["hi"]
    n_angles   = dim - 1

    def cart_quant(c, a, bits):
        levels = 1 << bits
        scale  = (levels - 1) / 2.0
        q   = np.clip(np.round(np.clip(c, -1.0, 1.0) * scale + scale / 2),
                      0, levels - 1).astype(np.int32)
        out = (q.astype(np.float32) + 0.5) / scale - 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-7, None)

    def ang_uniform(c, a, bits):
        return from_angles(_uquant(a, bits, hi))

    def ang_clip(c, a, keep_n, keep_bits, idx=None):
        out = clip_means[None, :].repeat(a.shape[0], axis=0)
        if keep_n > 0:
            _idx = idx[:keep_n] if idx is not None else np.arange(keep_n)
            out[:, _idx] = _uquant(a[:, _idx], keep_bits, hi[_idx])
        return from_angles(out)

    TIER_BITS = [0, 2, 4, 8, 16, 32]
    TIER_DIST = {
        0:  lambda r: r**2 / 12,
        2:  lambda r: (r / 4)**2 / 12,
        4:  lambda r: (r / 16)**2 / 12,
        8:  lambda r: (r / 256)**2 / 12,
        16: lambda r: (r * 2**-10)**2 / 12,
        32: lambda r: (r * 2**-23)**2 / 12,
    }

    def greedy_tiers(sens, budget):
        M      = len(sens)
        tiers  = np.zeros(M, dtype=np.int32)
        c_dist = np.array([sens[i] * TIER_DIST[0](float(hi[i]))
                           for i in range(M)])
        used   = 0
        def step(i):
            cur = tiers[i]
            if cur >= len(TIER_BITS) - 1: return None
            nxt = cur + 1
            xb  = TIER_BITS[nxt] - TIER_BITS[cur]
            nd  = sens[i] * TIER_DIST[TIER_BITS[nxt]](float(hi[i]))
            gain = c_dist[i] - nd
            return (gain / xb if xb > 0 else -np.inf, xb, nxt, nd)
        opts = [step(i) for i in range(M)]
        while used < budget:
            bi, br, bt = -1, -np.inf, None
            for i, o in enumerate(opts):
                if o is None or o[1] > budget - used: continue
                if o[0] > br: br, bi, bt = o[0], i, o
            if bi < 0 or br <= 0: break
            _, xb, new_tier, nd = bt
            tiers[bi] = new_tier; c_dist[bi] = nd; used += xb
            opts[bi] = step(bi)
        return tiers

    def apply_greedy(c, a, tiers):
        N, M = a.shape
        out  = clip_means[None, :].repeat(N, axis=0)
        for tid, bits in enumerate(TIER_BITS):
            idx = np.where(tiers == tid)[0]
            if not len(idx) or bits == 0: continue
            r = hi[idx]
            if bits <= 8:
                out[:, idx] = _uquant(a[:, idx], bits, r)
            elif bits == 16:
                out[:, idx] = a[:, idx].astype(np.float16).astype(np.float32)
            else:
                out[:, idx] = a[:, idx]
        return from_angles(out)

    # Build schemes
    sens = stats["sens"]
    schemes = []

    # Baselines
    schemes.append(("float32", lambda c, a: c, 32 * dim))
    for b in (8, 4, 2):
        schemes.append((f"cart_int{b}",
                        (lambda b_: lambda c, a: cart_quant(c, a, b_))(b),
                        b * dim))

    # Angle uniform
    for b in (8, 4, 2):
        schemes.append((f"angle_int{b}_uniform",
                        (lambda b_: lambda c, a: ang_uniform(c, a, b_))(b),
                        b * n_angles))

    # Clipping sweep (Jacobian-ranked) at fixed int8
    for frac in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8):
        keep_n = int(round(n_angles * (1.0 - frac)))
        bits   = keep_n * 8
        name   = f"jac_clip_{int(frac*100)}pct"
        schemes.append((name,
                        (lambda k_, jk_: lambda c, a: ang_clip(c, a, k_, 8, jk_)
                         )(keep_n, jac_rank),
                        bits))

    # Positional clipping (same fracs, for comparison)
    for frac in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8):
        keep_n = int(round(n_angles * (1.0 - frac)))
        bits   = keep_n * 8
        name   = f"pos_clip_{int(frac*100)}pct"
        schemes.append((name,
                        (lambda k_: lambda c, a: ang_clip(c, a, k_, 8))(keep_n),
                        bits))

    # Greedy tier budgets
    for budget in (1536, 3072, 6144, 9216, 12288):
        tiers    = greedy_tiers(sens, budget)
        actual   = int(sum(TIER_BITS[t] for t in tiers))
        name     = f"greedy_{budget}b"
        schemes.append((name,
                        (lambda t_: lambda c, a: apply_greedy(c, a, t_))(tiers),
                        actual))

    # Jacobian truncation sweep (int8/int4/int2)
    for keep_bits in (8, 4, 2):
        for frac in (0.3, 0.5, 0.7, 1.0):
            keep_n = int(round(n_angles * frac))
            bits   = keep_n * keep_bits
            name   = f"jac_trunc_int{keep_bits}_keep{int(frac*100)}pct"
            schemes.append((name,
                            (lambda k_, kb_, jk_:
                             lambda c, a: ang_clip(c, a, k_, kb_, jk_)
                             )(keep_n, keep_bits, jac_rank),
                            bits))

    return schemes


# ---------------------------------------------------------------------------
# Pass 2: streaming top-k accumulation for one scheme
# ---------------------------------------------------------------------------

def update_topk(top_scores: np.ndarray, top_gidx: np.ndarray,
                c_quant: np.ndarray, q_vecs: np.ndarray,
                gidx_start: int, k: int, q_batch: int) -> None:
    """In-place update of (top_scores, top_gidx) with scores against c_quant.

    Two-stage merge to keep the working set tiny:
      1. matmul -> sims (qb, B), then immediately argpartition down to local-k.
      2. merge the (qb, k) local result with the running (qb, k) state via a
         single argpartition on (qb, 2k).
    This avoids ever allocating a (qb, k+B) or (qb, B) int64 array and lets the
    big sims buffer be freed before the merge step.
    """
    n_q        = q_vecs.shape[0]
    batch_size = c_quant.shape[0]
    c_quant_T  = np.ascontiguousarray(c_quant.T)             # (D, B)

    if batch_size <= k:
        # Whole batch fits in top-k, no argpartition needed for stage 1
        new_gidx_full = (gidx_start +
                         np.arange(batch_size, dtype=np.int64))   # (B,)
        for qs in range(0, n_q, q_batch):
            qe   = min(qs + q_batch, n_q)
            sims = q_vecs[qs:qe] @ c_quant_T                      # (qb, B)
            local_g = np.broadcast_to(new_gidx_full,
                                      (qe - qs, batch_size))
            merged_s = np.concatenate([top_scores[qs:qe], sims], axis=1)
            merged_g = np.concatenate([top_gidx[qs:qe], local_g], axis=1)
            sel = np.argpartition(-merged_s, kth=k - 1, axis=1)[:, :k]
            top_scores[qs:qe] = np.take_along_axis(merged_s, sel, axis=1)
            top_gidx[qs:qe]   = np.take_along_axis(merged_g, sel, axis=1)
        return

    for qs in range(0, n_q, q_batch):
        qe = min(qs + q_batch, n_q)

        # Stage 1: matmul + local top-k
        sims      = q_vecs[qs:qe] @ c_quant_T                     # (qb, B) f32
        local_sel = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]   # (qb, k)
        local_s   = np.take_along_axis(sims, local_sel, axis=1)        # (qb, k)
        del sims                                                       # free B-sized buf
        local_g   = (gidx_start + local_sel).astype(np.int64, copy=False)
        del local_sel

        # Stage 2: merge running (qb, k) state with local (qb, k) -> (qb, 2k)
        merged_s = np.concatenate([top_scores[qs:qe], local_s], axis=1)  # (qb, 2k)
        merged_g = np.concatenate([top_gidx[qs:qe], local_g], axis=1)
        del local_s, local_g
        sel = np.argpartition(-merged_s, kth=k - 1, axis=1)[:, :k]
        top_scores[qs:qe] = np.take_along_axis(merged_s, sel, axis=1)
        top_gidx[qs:qe]   = np.take_along_axis(merged_g, sel, axis=1)
        del merged_s, merged_g, sel


def _eval_metrics(top_scores: np.ndarray, top_gidx: np.ndarray,
                  q_ids: list[str], doc_id_arr: np.ndarray,
                  qrels: dict[str, dict[str, int]]) -> tuple[float, float, int]:
    ndcg_sc: list[float] = []
    rec_sc:  list[float] = []
    for qi, qid in enumerate(q_ids):
        rel = qrels.get(qid)
        if not rel:
            continue
        order      = np.argsort(-top_scores[qi])
        ranked_idx = top_gidx[qi][order]
        ranked_ids = doc_id_arr[ranked_idx]
        rels  = [rel.get(did, 0) for did in ranked_ids[:10]]
        ideal = sorted(rel.values(), reverse=True)
        idcg  = sum(r / math.log2(i + 2) for i, r in enumerate(ideal[:10]))
        ndcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rels)) / idcg \
                if idcg else 0.0
        ndcg_sc.append(ndcg)
        hits = sum(1 for did in ranked_ids[:100] if did in rel)
        rec_sc.append(hits / len(rel))
    return (float(np.mean(ndcg_sc)) if ndcg_sc else 0.0,
            float(np.mean(rec_sc))   if rec_sc  else 0.0,
            len(ndcg_sc))


def _build_super_batches(corpus_chunks: list[Path],
                         target_docs: int) -> list[tuple[list[Path], int]]:
    """Group consecutive chunks into super-batches summing to ~target_docs.
    Reads only the .npy header (mmap_mode='r') for sizes — no data load."""
    groups: list[tuple[list[Path], int]] = []
    cur, cur_n = [], 0
    for cp in corpus_chunks:
        s = np.load(cp, mmap_mode="r").shape[0]
        cur.append(cp)
        cur_n += s
        if cur_n >= target_docs:
            groups.append((cur, cur_n))
            cur, cur_n = [], 0
    if cur:
        groups.append((cur, cur_n))
    return groups


def score_all_schemes(
    schemes: list[tuple[str, object, int]],
    corpus_chunks: list[Path],
    q_vecs: np.ndarray,
    q_ids: list[str],
    all_doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    progress_dir: Path,
    k: int = 100,
    q_batch: int = 8192,
    checkpoint_every: int = 50,
    super_batch_docs: int = 50_000,
    scheme_workers: int = 1,
) -> list[dict]:
    """Chunk-major scoring with super-batching for BLAS efficiency.

    Many small disk chunks are concatenated into a single ~super_batch_docs-row
    matrix before scoring, so each (n_q, D) @ (D, B) matmul is large enough to
    saturate multi-core BLAS. All schemes run against the same super-batch in
    one pass (angles computed once).
    """
    progress_dir.mkdir(parents=True, exist_ok=True)
    state_path = progress_dir / "scoring_state.npz"
    n_schemes  = len(schemes)
    n_q        = len(q_ids)
    scheme_names = [s[0] for s in schemes]

    print(f"  building super-batches (~{super_batch_docs:,} docs each) ...",
          flush=True)
    super_batches = _build_super_batches(corpus_chunks, super_batch_docs)
    n_groups      = len(super_batches)
    print(f"  {n_groups} super-batches from {len(corpus_chunks)} chunks", flush=True)

    if state_path.exists():
        d = np.load(state_path, allow_pickle=True)
        saved_names = list(d["scheme_names"])
        saved_sb    = int(d["super_batch_docs"]) if "super_batch_docs" in d.files else -1
        if saved_names != scheme_names:
            raise ValueError(
                f"Scheme list changed since checkpoint.\n"
                f"  saved:   {saved_names}\n  current: {scheme_names}\n"
                f"Delete {state_path} and restart, or pass --schemes to match.")
        if saved_sb != super_batch_docs:
            raise ValueError(
                f"--super-batch changed ({saved_sb} -> {super_batch_docs}). "
                f"Delete {state_path} and restart, or rerun with the original "
                f"value.")
        top_scores    = d["top_scores"].astype(np.float32, copy=False)
        top_gidx      = d["top_gidx"].astype(np.int64, copy=False)
        groups_done   = int(d["groups_done"])
        global_offset = int(d["global_offset"])
        print(f"  resuming from super-batch {groups_done}/{n_groups}  "
              f"({global_offset:,} docs processed)")
    else:
        top_scores    = np.full((n_schemes, n_q, k), -np.inf, dtype=np.float32)
        top_gidx      = np.zeros((n_schemes, n_q, k), dtype=np.int64)
        groups_done   = 0
        global_offset = 0

    t0            = time.perf_counter()
    docs_at_start = global_offset
    last_print    = 0.0

    # Optional thread pool for running schemes concurrently within a super-batch.
    # Each worker calls scheme_fn (Python/numpy) and update_topk (BLAS matmul +
    # argpartition). top_scores[si] / top_gidx[si] slices are disjoint per
    # scheme, so concurrent writes are race-free.
    executor = None
    if scheme_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=scheme_workers,
                                      thread_name_prefix="scheme")
        print(f"  scheme parallelism: {scheme_workers} workers", flush=True)

    profile_first = True
    for gi, (group_paths, group_size) in enumerate(super_batches):
        if gi < groups_done:
            continue

        prof = profile_first
        t_load = t_ang = t_score = 0.0

        # Load + concatenate this super-batch
        ts = time.perf_counter()
        if len(group_paths) == 1:
            c_batch, _ = load_chunk(group_paths[0])
        else:
            arrs = [load_chunk(p)[0] for p in group_paths]
            c_batch = np.concatenate(arrs, axis=0).astype(np.float32, copy=False)
            del arrs
        if prof: t_load = time.perf_counter() - ts

        ts = time.perf_counter()
        c_angles = to_angles(c_batch)
        if prof: t_ang = time.perf_counter() - ts
        bsize    = c_batch.shape[0]

        def _run_scheme(si_fn):
            si, sfn = si_fn
            c_quant = sfn(c_batch, c_angles)
            update_topk(top_scores[si], top_gidx[si], c_quant, q_vecs,
                        global_offset, k, q_batch)

        ts = time.perf_counter()
        tasks = [(si, sfn) for si, (_n, sfn, _b) in enumerate(schemes)]
        if executor is not None:
            list(executor.map(_run_scheme, tasks))
        else:
            for t in tasks:
                _run_scheme(t)
        if prof: t_score = time.perf_counter() - ts

        if prof:
            mode = (f"{scheme_workers}-way parallel"
                    if executor is not None else "sequential")
            print(f"  [profile sb #1, {bsize:,} docs, {len(schemes)} schemes, {mode}]\n"
                  f"    load        = {t_load*1000:>7.0f} ms\n"
                  f"    to_angles   = {t_ang*1000:>7.0f} ms\n"
                  f"    all schemes = {t_score*1000:>7.0f} ms  "
                  f"(wall avg {t_score/len(schemes)*1000:.0f} ms/scheme)\n"
                  f"    total       = {(t_load+t_ang+t_score)*1000:>7.0f} ms",
                  flush=True)
            profile_first = False

        global_offset += bsize
        groups_done    = gi + 1
        del c_batch, c_angles
        free_mem()

        if groups_done % checkpoint_every == 0 or groups_done == n_groups:
            tmp = state_path.parent / (state_path.name + ".tmp")
            with tmp.open("wb") as f:
                np.savez(f,
                         top_scores=top_scores, top_gidx=top_gidx,
                         groups_done=np.int64(groups_done),
                         global_offset=np.int64(global_offset),
                         super_batch_docs=np.int64(super_batch_docs),
                         scheme_names=np.array(scheme_names, dtype=object))
            os.replace(tmp, state_path)

        now = time.perf_counter()
        if now - last_print > 5 or groups_done == n_groups:
            elapsed = now - t0
            done    = global_offset - docs_at_start
            rate    = done / max(elapsed, 1e-3)
            remain  = (len(all_doc_ids) - global_offset) / max(rate, 1)
            print(f"  super-batch {groups_done}/{n_groups}  "
                  f"{global_offset:,} docs  {n_schemes} schemes  "
                  f"{rate:.0f} doc/s  ETA {remain/60:.1f}min   ",
                  end="\r", flush=True)
            last_print = now

    print()

    if executor is not None:
        executor.shutdown(wait=True)

    doc_id_arr = np.array(all_doc_ids, dtype=object)
    results: list[dict] = []
    elapsed_total = time.perf_counter() - t0
    for si, (sname, _sfn, bits) in enumerate(schemes):
        ndcg, rec, nq = _eval_metrics(top_scores[si], top_gidx[si],
                                      q_ids, doc_id_arr, qrels)
        results.append({
            "scheme":       sname,
            "bits_per_vec": bits,
            "ndcg10":       ndcg,
            "recall100":    rec,
            "n_queries":    nq,
            "n_corpus":     global_offset,
            "elapsed_s":    elapsed_total,
        })
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(results: list[dict], base_bits: int, path: Path) -> None:
    hdr = (f"{'scheme':<45} {'bits':>8} {'ratio':>7} "
           f"{'NDCG@10':>9} {'R@100':>8} {'queries':>9} {'corpus':>12}")
    sep = "-" * len(hdr)
    lines = [hdr, sep]
    for r in results:
        ratio = base_bits / r["bits_per_vec"] if r["bits_per_vec"] > 0 else float("inf")
        lines.append(
            f"{r['scheme']:<45} {r['bits_per_vec']:>8d} {ratio:>7.1f}x "
            f"{r['ndcg10']:>9.4f} {r['recall100']:>8.4f} "
            f"{r['n_queries']:>9,} {r['n_corpus']:>12,}")
    text = "\n".join(lines)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) + ["compression_ratio"])
        w.writeheader()
        for r in results:
            row = dict(r)
            row["compression_ratio"] = (base_bits / r["bits_per_vec"]
                                        if r["bits_per_vec"] > 0 else float("inf"))
            w.writerow(row)
    print(f"results -> {path}  (and .csv)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir",  required=True,
                    help="Root with corpus/ and queries/ subdirs of chunked .npy files.")
    ap.add_argument("--data-dir",   required=True,
                    help="BEIR dataset dir (contains qrels/).")
    ap.add_argument("--embed-tag",  default="",
                    help='Chunk filename tag ("" for plain OAI layout).')
    ap.add_argument("--output-dir", default="eval_results_full")
    ap.add_argument("--super-batch", type=int, default=50_000,
                    help="Concatenate consecutive disk chunks until reaching "
                         "~N docs, then score that whole block in one BLAS "
                         "matmul. Bigger = faster on multi-core boxes, more "
                         "RAM. Default 50k (good for 96-core c5a.24xlarge).")
    ap.add_argument("--q-batch",    type=int, default=1024,
                    help="Query batch size for scoring matmul (default 1024). "
                         "Smaller = better cache locality for argpartition; "
                         "larger = bigger BLAS calls.")
    ap.add_argument("--k",          type=int, default=100,
                    help="Top-k to accumulate (default 100).")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="Save scoring state every N super-batches (default 10).")
    ap.add_argument("--blas-threads", type=int, default=0,
                    help="Force BLAS thread count per call (0 = auto: "
                         "cpu_count // scheme_workers).")
    ap.add_argument("--scheme-workers", type=int, default=0,
                    help="Run N schemes concurrently per super-batch "
                         "(0 = auto, 1 = sequential). Each worker gets its "
                         "own argpartition / from_angles thread; BLAS threads "
                         "are split across workers to avoid oversubscription.")
    ap.add_argument("--schemes",    nargs="*", default=None,
                    help="Scheme names to run (default: all). "
                         "Use --list-schemes to see names.")
    ap.add_argument("--list-schemes", action="store_true",
                    help="Print all scheme names and exit.")
    args = ap.parse_args()

    n_cpu = os.cpu_count() or 1
    if args.scheme_workers == 0:
        # Auto: aim for ~12 BLAS threads per worker (good matmul efficiency
        # on c5a-class boxes). Capped by cpu_count and by 8 workers.
        scheme_workers = max(1, min(8, n_cpu // 12))
    else:
        scheme_workers = args.scheme_workers

    if args.blas_threads:
        blas_threads = args.blas_threads
    else:
        blas_threads = max(1, n_cpu // scheme_workers)

    print(f"Configuring BLAS: {blas_threads} threads/call  "
          f"({scheme_workers} scheme workers x {blas_threads} = "
          f"{scheme_workers * blas_threads} of {n_cpu} cores)")
    configure_blas(blas_threads)

    embed_dir  = Path(args.embed_dir).expanduser().resolve()
    data_dir   = Path(args.data_dir).expanduser().resolve()
    out_dir    = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_root = out_dir / "progress"
    progress_root.mkdir(exist_ok=True)

    # Discover corpus chunks
    corpus_dir    = embed_dir / "corpus"
    corpus_chunks = discover_chunks(corpus_dir, args.embed_tag)
    if not corpus_chunks:
        raise FileNotFoundError(
            f"No corpus chunks in {corpus_dir} with tag='{args.embed_tag}'")
    print(f"Found {len(corpus_chunks)} corpus chunks")

    # Load queries (fits in RAM)
    print("Loading query embeddings ...")
    q_vecs, q_ids = load_queries(embed_dir / "queries", args.embed_tag)
    print(f"  {q_vecs.shape[0]:,} queries  dim={q_vecs.shape[1]}")

    dim = q_vecs.shape[1]

    # Load qrels and filter to queries that have any relevant doc
    print("Loading qrels ...")
    qrels = load_qrels(data_dir)
    q_with_qrels = [qid for qid in q_ids if qid in qrels]
    print(f"  {len(q_with_qrels):,} / {len(q_ids):,} queries have qrels")

    qid_idx  = {qid: i for i, qid in enumerate(q_ids)}
    q_filt   = q_vecs[[qid_idx[qid] for qid in q_with_qrels]]
    qrels_f  = {qid: qrels[qid] for qid in q_with_qrels}

    # Build flat doc ID list (needed to map global indices → doc IDs at eval time)
    # We read chunk .ids.txt files without loading the vectors
    print("Reading corpus doc IDs ...")
    all_doc_ids: list[str] = []
    for cp in corpus_chunks:
        stem  = cp.name[:-len(".npy")]
        ids_p = cp.parent / (stem + ".ids.txt")
        with ids_p.open("r", encoding="utf-8") as f:
            all_doc_ids.extend(ln.rstrip("\n") for ln in f)
    print(f"  {len(all_doc_ids):,} total corpus docs")

    # Pass 1: stats
    print("\n=== Pass 1: collecting stats from full corpus ===")
    stats = collect_stats(
        corpus_chunks,
        stats_path   = progress_root / "stats.npz",
        partial_path = progress_root / "stats_partial.npz",
        dim          = dim,
    )

    # Build schemes
    schemes = make_schemes(stats, dim)
    if args.list_schemes:
        for name, _, bits in schemes:
            print(f"  {name:<45} {bits:>8}b")
        return

    if args.schemes:
        schemes = [(n, fn, b) for n, fn, b in schemes if n in args.schemes]
        if not schemes:
            print("ERROR: no matching schemes found. Use --list-schemes.")
            return

    base_bits = 32 * dim
    print(f"\n=== Pass 2: scoring {len(schemes)} schemes "
          f"against {len(all_doc_ids):,} docs ===")
    print(f"  float32 baseline = {base_bits:,} bits/vec")
    print(f"  corpus chunks    = {len(corpus_chunks)}")
    print(f"  super-batch={args.super_batch:,}  q-batch={args.q_batch}  "
          f"checkpoint every {args.checkpoint_every} super-batches\n")

    # Friendly notice about leftover per-scheme dirs from the old layout.
    legacy = [p for p in progress_root.iterdir()
              if p.is_dir() and ((p / "state.json").exists() or (p / "DONE").exists())]
    if legacy:
        print(f"  [info] ignoring {len(legacy)} stale per-scheme progress dirs "
              f"(old layout). Safe to delete: {progress_root}/{{{','.join(p.name for p in legacy[:3])}{',...' if len(legacy)>3 else ''}}}\n")

    results = score_all_schemes(
        schemes          = schemes,
        corpus_chunks    = corpus_chunks,
        q_vecs           = q_filt,
        q_ids            = q_with_qrels,
        all_doc_ids      = all_doc_ids,
        qrels            = qrels_f,
        progress_dir     = progress_root,
        k                = args.k,
        q_batch          = args.q_batch,
        checkpoint_every = args.checkpoint_every,
        super_batch_docs = args.super_batch,
        scheme_workers   = scheme_workers,
    )

    write_results(results, base_bits, out_dir / "results_streaming.txt")
    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
