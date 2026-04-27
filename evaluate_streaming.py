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
  Pass 2 (score):  for each scheme, saves {scheme}/top_scores.npy,
                   {scheme}/top_gidx.npy, and {scheme}/state.json after each
                   corpus batch. Restart the script to resume from the last
                   saved batch. Completed schemes (DONE marker) are skipped.

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
    x    = x.astype(np.float32, copy=False)
    N, n = x.shape
    out  = np.zeros((N, n - 1), dtype=np.float32)
    sq   = x ** 2
    tail = np.sqrt(np.clip(np.cumsum(sq[:, ::-1], axis=1)[:, ::-1], 0.0, None))
    del sq
    for i in range(n - 2):
        out[:, i] = np.arccos(np.clip(x[:, i] / np.clip(tail[:, i], 1e-7, None),
                                      -1.0, 1.0))
    last = np.arctan2(x[:, n - 1], x[:, n - 2])
    out[:, n - 2] = np.where(last < 0, last + 2 * math.pi, last)
    del tail, last
    return out


def from_angles(a: np.ndarray) -> np.ndarray:
    N, m = a.shape
    x    = np.zeros((N, m + 1), dtype=np.float32)
    sp   = np.ones(N, dtype=np.float32)
    for i in range(m):
        x[:, i] = sp * np.cos(a[:, i])
        sp = sp * np.sin(a[:, i])
    x[:, m] = sp
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
    Queries are batched to keep the score matrix small."""
    n_q        = q_vecs.shape[0]
    batch_size = c_quant.shape[0]
    new_gidx   = np.arange(gidx_start, gidx_start + batch_size, dtype=np.int64)

    for qs in range(0, n_q, q_batch):
        qe     = min(qs + q_batch, n_q)
        sims   = q_vecs[qs:qe] @ c_quant.T                         # (qb, B)

        comb_s = np.concatenate([top_scores[qs:qe], sims], axis=1) # (qb, k+B)
        comb_g = np.concatenate(
            [top_gidx[qs:qe],
             np.broadcast_to(new_gidx, (qe - qs, batch_size)).copy()],
            axis=1)                                                 # (qb, k+B)

        kth    = min(k, comb_s.shape[1] - 1)
        sel    = np.argpartition(-comb_s, kth=kth, axis=1)[:, :k]

        top_scores[qs:qe] = np.take_along_axis(comb_s, sel, axis=1)
        top_gidx[qs:qe]   = np.take_along_axis(comb_g, sel, axis=1)
        del sims, comb_s, comb_g, sel


def score_scheme(
    scheme_name: str,
    scheme_fn,
    bits_per_vec: int,
    corpus_chunks: list[Path],
    q_vecs: np.ndarray,
    q_ids: list[str],
    all_doc_ids: list[str],    # flat list in chunk order
    qrels: dict[str, dict[str, int]],
    progress_dir: Path,
    k: int = 100,
    q_batch: int = 512,
) -> dict:
    """Score all corpus chunks for one scheme. Resumable."""
    progress_dir.mkdir(parents=True, exist_ok=True)
    done_marker  = progress_dir / "DONE"
    state_path   = progress_dir / "state.json"
    ts_path      = progress_dir / "top_scores.npy"
    tg_path      = progress_dir / "top_gidx.npy"

    n_q = len(q_ids)

    if done_marker.exists():
        print(f"  [{scheme_name}] already complete, loading saved results")
        top_scores = np.load(ts_path)
        top_gidx   = np.load(tg_path)
        chunks_done = len(corpus_chunks)
        global_offset = len(all_doc_ids)
    elif state_path.exists():
        st = json.loads(state_path.read_text())
        chunks_done   = st["chunks_done"]
        global_offset = st["global_offset"]
        top_scores    = np.load(ts_path)
        top_gidx      = np.load(tg_path)
        print(f"  [{scheme_name}] resuming from chunk {chunks_done}  "
              f"({global_offset:,} docs processed)")
    else:
        chunks_done   = 0
        global_offset = 0
        top_scores    = np.full((n_q, k), -np.inf, dtype=np.float32)
        top_gidx      = np.zeros((n_q, k), dtype=np.int64)

    t0 = time.perf_counter()
    n_total = len(corpus_chunks)

    for ci, chunk_path in enumerate(corpus_chunks):
        if ci < chunks_done:
            # We still need to advance global_offset -- read just the size
            if global_offset == 0:   # first resume pass, recalculate offset
                for prev_ci in range(ci + 1):
                    s = np.load(corpus_chunks[prev_ci], mmap_mode="r").shape[0]
                    global_offset += s
            continue

        c_batch, _ = load_chunk(chunk_path)
        c_angles   = to_angles(c_batch)
        c_quant    = scheme_fn(c_batch, c_angles)
        del c_batch, c_angles
        free_mem()

        update_topk(top_scores, top_gidx, c_quant, q_vecs,
                    global_offset, k, q_batch)
        global_offset += c_quant.shape[0]
        del c_quant
        free_mem()

        chunks_done = ci + 1
        # Atomic save: write then rename (open file handle avoids np.save
        # appending an extra .npy to paths that already end in .npy.tmp)
        tmp_ts = ts_path.parent / (ts_path.name + ".tmp")
        tmp_tg = tg_path.parent / (tg_path.name + ".tmp")
        with tmp_ts.open("wb") as f: np.save(f, top_scores)
        with tmp_tg.open("wb") as f: np.save(f, top_gidx)
        os.replace(tmp_ts, ts_path)
        os.replace(tmp_tg, tg_path)
        state_path.write_text(json.dumps(
            {"chunks_done": chunks_done, "global_offset": global_offset}))

        elapsed = time.perf_counter() - t0
        rate    = global_offset / elapsed
        remain  = (len(all_doc_ids) - global_offset) / max(rate, 1)
        print(f"  [{scheme_name}] {chunks_done}/{n_total}  "
              f"{global_offset:,} docs  {rate:.0f} doc/s  "
              f"ETA {remain/60:.0f}min", end="\r")

    print()
    done_marker.write_text("ok\n")

    # Evaluate NDCG@10 and Recall@100 from top_gidx
    # dtype=object stores Python string pointers (~43 MB) vs dtype=str which
    # pads every string to max length and can exceed 4 GB on FEVER.
    doc_id_arr = np.array(all_doc_ids, dtype=object)
    ndcg_sc: list[float] = []
    rec_sc:  list[float] = []

    for qi, qid in enumerate(q_ids):
        rel = qrels.get(qid)
        if not rel:
            continue
        # Sort top_gidx[qi] by top_scores[qi] descending
        order  = np.argsort(-top_scores[qi])
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

    elapsed = time.perf_counter() - t0
    return {
        "scheme":       scheme_name,
        "bits_per_vec": bits_per_vec,
        "ndcg10":       float(np.mean(ndcg_sc))  if ndcg_sc else 0.0,
        "recall100":    float(np.mean(rec_sc))    if rec_sc  else 0.0,
        "n_queries":    len(ndcg_sc),
        "n_corpus":     global_offset,
        "elapsed_s":    elapsed,
    }


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
    ap.add_argument("--c-batch",    type=int, default=30_000,
                    help="Corpus docs per batch (default 30k).")
    ap.add_argument("--q-batch",    type=int, default=512,
                    help="Query batch size for scoring (default 512).")
    ap.add_argument("--k",          type=int, default=100,
                    help="Top-k to accumulate (default 100).")
    ap.add_argument("--schemes",    nargs="*", default=None,
                    help="Scheme names to run (default: all). "
                         "Use --list-schemes to see names.")
    ap.add_argument("--list-schemes", action="store_true",
                    help="Print all scheme names and exit.")
    args = ap.parse_args()

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
    print(f"  corpus batches   = {len(corpus_chunks)}\n")

    results: list[dict] = []
    for scheme_name, scheme_fn, bits in schemes:
        print(f"\n--- scheme: {scheme_name}  ({bits}b/vec, "
              f"{base_bits/bits:.1f}x compression) ---")
        r = score_scheme(
            scheme_name  = scheme_name,
            scheme_fn    = scheme_fn,
            bits_per_vec = bits,
            corpus_chunks= corpus_chunks,
            q_vecs       = q_filt,
            q_ids        = q_with_qrels,
            all_doc_ids  = all_doc_ids,
            qrels        = qrels_f,
            progress_dir = progress_root / scheme_name,
            k            = args.k,
            q_batch      = args.q_batch,
        )
        results.append(r)
        print(f"  NDCG@10={r['ndcg10']:.4f}  "
              f"R@100={r['recall100']:.4f}  "
              f"{r['elapsed_s']:.0f}s")

    write_results(results, base_bits, out_dir / "results_streaming.txt")
    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
