"""
Quantization evaluation on real BEIR embeddings.

Addresses four questions:
  1. Accuracy:     angle-quantization vs Cartesian-int8 -- which is better
                   at the same bit budget?
  2. int4 / int2:  is the quality gap over int8 significant?
  3. Clipping:     what does "0-bit" (store only corpus mean) cost in quality?
                   validates that CLIPPED angles carry negligible information.
  4. Tradeoff:     generates a Compression-ratio vs NDCG@10 plot (PNG + CSV).

Inputs:
  --embed-dir   root of chunked embedding output (e.g. embeddings/fever)
                expects sub-dirs corpus/ and queries/ with chunk_*.npy files.
  --data-dir    BEIR dataset directory (e.g. bier-data/fever)
                reads qrels/test.tsv (falls back to dev.tsv then train.tsv).
  --embed-tag   filename tag to glob for (e.g. "" for OAI, "minilm-l6-vllm")

Memory:
  Full FEVER corpus (5.4M x 1536 x fp32) = ~33 GB. Use --max-corpus to
  sample a manageable subset for machines with limited RAM (default 200_000).
  Queries with no relevant doc in the sample are dropped automatically.

Outputs:
  results_table.txt    -- human-readable metrics table
  results.csv          -- machine-readable metrics
  tradeoff.png         -- compression ratio vs NDCG@10 plot
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def _glob_chunks(split_dir: Path, tag: str) -> list[Path]:
    if tag:
        chunks = sorted(split_dir.glob(f"chunk_*.{tag}.npy"))
    else:
        # no tag -- original OAI layout: chunk_XXXXXXX.npy (no extra dot-segment)
        chunks = sorted(p for p in split_dir.glob("chunk_*.npy")
                        if p.name.count(".") == 1)
    return chunks


def load_split(split_dir: Path, tag: str,
               max_vecs: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Load all chunks for a split. Returns (vecs float32, ids list)."""
    chunks = _glob_chunks(split_dir, tag)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks found in {split_dir} with tag='{tag}'. "
            f"Check --embed-tag and that embeddings exist.")

    all_vecs: list[np.ndarray] = []
    all_ids:  list[str] = []
    for p in chunks:
        arr = np.load(p)
        stem = p.name[:-len(".npy")]
        ids_p = split_dir / (stem + ".ids.txt")
        with ids_p.open("r", encoding="utf-8") as f:
            ids = [ln.rstrip("\n") for ln in f]
        assert arr.shape[0] == len(ids), f"shape mismatch in {p.name}"
        all_vecs.append(arr)
        all_ids.extend(ids)
        if max_vecs and sum(v.shape[0] for v in all_vecs) >= max_vecs:
            break

    vecs = np.concatenate(all_vecs, axis=0).astype(np.float32)
    if max_vecs:
        vecs = vecs[:max_vecs]
        all_ids = all_ids[:max_vecs]
    return vecs, all_ids


# ---------------------------------------------------------------------------
# Qrels loading
# ---------------------------------------------------------------------------

def load_qrels(data_dir: Path) -> dict[str, dict[str, int]]:
    """Returns {query_id: {corpus_id: relevance_score}}."""
    qrels_dir = data_dir / "qrels"
    for name in ("test.tsv", "dev.tsv", "train.tsv"):
        p = qrels_dir / name
        if p.exists():
            print(f"loading qrels from {p}")
            break
    else:
        raise FileNotFoundError(f"No qrels file found in {qrels_dir}")

    qrels: dict[str, dict[str, int]] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid  = row["query-id"]
            cid  = row["corpus-id"]
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(qid, {})[cid] = score
    return qrels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dcg(relevances: list[int], k: int) -> float:
    s = 0.0
    for i, r in enumerate(relevances[:k], start=1):
        s += r / math.log2(i + 1)
    return s


def ndcg_at_k(top_ids: np.ndarray, relevant: dict[str, int], k: int) -> float:
    rels = [relevant.get(cid, 0) for cid in top_ids[:k]]
    ideal = sorted(relevant.values(), reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(rels, k) / idcg


def recall_at_k(top_ids: np.ndarray, relevant: dict[str, int], k: int) -> float:
    hits = sum(1 for cid in top_ids[:k] if cid in relevant)
    return hits / len(relevant) if relevant else 0.0


@dataclass
class EvalResult:
    label: str
    bits_per_vec: int
    dim: int
    ndcg10: float
    recall100: float
    score_ms: float       # total scoring time
    family: str = ""      # for plot grouping


def evaluate(
    label: str,
    bits_per_vec: int,
    q_vecs: np.ndarray,         # (Q, d) unit
    c_vecs: np.ndarray,         # (C, d) unit
    q_ids: list[str],
    c_ids: list[str],
    qrels: dict[str, dict[str, int]],
    k_ndcg: int = 10,
    k_recall: int = 100,
    batch_q: int = 512,
    family: str = "",
) -> EvalResult:
    """Score queries in batches; compute NDCG@10 and Recall@100."""
    c_ids_arr = np.array(c_ids)
    top_k = max(k_ndcg, k_recall)

    ndcg_scores:   list[float] = []
    recall_scores: list[float] = []
    t0 = time.perf_counter()

    for qi in range(0, len(q_ids), batch_q):
        q_batch = q_vecs[qi: qi + batch_q]          # (B, d)
        sims = q_batch @ c_vecs.T                    # (B, C)
        top_idx = np.argpartition(-sims, kth=min(top_k, sims.shape[1] - 1),
                                  axis=1)[:, :top_k]
        # sort each row
        for bi in range(q_batch.shape[0]):
            qid = q_ids[qi + bi]
            relevant = qrels.get(qid)
            if not relevant:
                continue
            row_sims = sims[bi, top_idx[bi]]
            order = np.argsort(-row_sims)
            ranked_ids = c_ids_arr[top_idx[bi][order]]
            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant, k_ndcg))
            recall_scores.append(recall_at_k(ranked_ids, relevant, k_recall))

    elapsed = time.perf_counter() - t0
    return EvalResult(
        label=label,
        bits_per_vec=bits_per_vec,
        dim=q_vecs.shape[1],
        ndcg10=float(np.mean(ndcg_scores))   if ndcg_scores   else 0.0,
        recall100=float(np.mean(recall_scores)) if recall_scores else 0.0,
        score_ms=elapsed * 1000,
        family=family,
    )


# ---------------------------------------------------------------------------
# Quantization helpers (self-contained, no imports from geo*.py)
# ---------------------------------------------------------------------------

# -- Cartesian scalar quantization --

def cartesian_quant(vecs: np.ndarray, bits: int) -> np.ndarray:
    """Uniform scalar quantization of Cartesian unit vectors. Returns float32."""
    levels = (1 << bits)
    scale  = (levels - 1) / 2.0           # maps [-1, 1] -> [0, levels-1]
    q = np.round(np.clip(vecs, -1.0, 1.0) * scale).astype(np.int32)
    q = np.clip(q, 0, levels - 1)
    out = (q.astype(np.float32) / scale) - 1.0  # not quite right; use midpoint below
    # midpoint dequant
    out = (q.astype(np.float32) + 0.5) / scale - 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)


# -- Hyperspherical conversion --

def cartesian_to_hyperspherical(x: np.ndarray) -> np.ndarray:
    """float32 throughout -- halves peak memory vs float64 with negligible precision loss."""
    x = x.astype(np.float32, copy=False)
    N, n = x.shape
    angles = np.zeros((N, n - 1), dtype=np.float32)
    sq = x ** 2
    rev_cumsum = np.cumsum(sq[:, ::-1], axis=1)[:, ::-1]
    tail = np.sqrt(np.clip(rev_cumsum, 0.0, None))
    del sq, rev_cumsum
    for i in range(n - 2):
        denom = np.clip(tail[:, i], 1e-7, None)
        ratio = np.clip(x[:, i] / denom, -1.0, 1.0)
        angles[:, i] = np.arccos(ratio)
    last = np.arctan2(x[:, n - 1], x[:, n - 2])
    angles[:, n - 2] = np.where(last < 0, last + 2 * math.pi, last)
    del tail, last
    return angles


def hyperspherical_to_cartesian(angles: np.ndarray) -> np.ndarray:
    angles = angles.astype(np.float32, copy=False)
    N, m = angles.shape
    n = m + 1
    x = np.zeros((N, n), dtype=np.float32)
    sin_prod = np.ones(N, dtype=np.float32)
    for i in range(m):
        x[:, i] = sin_prod * np.cos(angles[:, i])
        sin_prod = sin_prod * np.sin(angles[:, i])
    x[:, n - 1] = sin_prod
    return x


def recon_from_angles(angles: np.ndarray) -> np.ndarray:
    x = hyperspherical_to_cartesian(angles).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


# -- Uniform angle quantization --

def angle_uniform_quant(angles: np.ndarray, bits_per_angle: int) -> np.ndarray:
    M  = angles.shape[1]
    hi = np.full(M, math.pi, dtype=np.float32)
    hi[-1] = 2 * math.pi
    levels = 1 << bits_per_angle
    q = np.floor(np.clip(angles, 0.0, hi - 1e-6) / hi * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)
    return (q.astype(np.float32) + 0.5) * (hi / levels)


# -- Jacobian-aware bit allocation (from geometry.py) --

def jacobian_bit_allocation(n_angles: int, total_bits: int,
                             empirical_std: np.ndarray | None = None,
                             min_bits: int = 0, max_bits: int = 16) -> np.ndarray:
    if empirical_std is not None:
        log_std = np.log2(np.clip(empirical_std, 1e-6, None))
        importance = log_std - log_std.min() + 0.1
    else:
        importance = np.array([math.sqrt(n_angles - i) for i in range(n_angles)])

    weights = importance / importance.sum()
    ideal   = weights * total_bits
    bits    = np.clip(np.floor(ideal).astype(np.int32), min_bits, max_bits)
    remaining = int(total_bits - bits.sum())
    guard = 0
    max_iters = n_angles * max_bits * 4
    while remaining != 0 and guard < max_iters:
        residual = ideal - bits
        if remaining > 0:
            mask = bits < max_bits
            if not mask.any():
                break
            j = int(np.argmax(np.where(mask, residual, -np.inf)))
            bits[j] += 1; remaining -= 1
        else:
            mask = bits > min_bits
            if not mask.any():
                break
            j = int(np.argmin(np.where(mask, residual, np.inf)))
            bits[j] -= 1; remaining += 1
        guard += 1
    return bits


def angle_jacobian_quant(angles: np.ndarray, bits_arr: np.ndarray) -> np.ndarray:
    M  = angles.shape[1]
    hi = np.full(M, math.pi, dtype=np.float32)
    hi[-1] = 2 * math.pi
    levels = (1 << bits_arr.astype(np.int32))
    a = np.clip(angles, 0.0, hi - 1e-6)
    q = np.clip(np.floor(a / hi * levels).astype(np.int32), 0, levels - 1)
    del a
    deq = np.zeros_like(angles, dtype=np.float32)
    for i in range(M):
        if bits_arr[i] == 0:
            deq[:, i] = hi[i] / 2.0
        else:
            deq[:, i] = (q[:, i].astype(np.float32) + 0.5) * (hi[i] / levels[i])
    del q
    return deq


# -- Dynamic tier assignment (from geo2.py) --

TIER_BITS_ARR = [0, 2, 4, 8, 16, 32]   # indexed by tier id 0..5

def estimate_sensitivity(angles_calib: np.ndarray) -> np.ndarray:
    a = angles_calib.astype(np.float32, copy=False)
    log_sin2 = 2.0 * np.log(np.clip(np.sin(a), 1e-7, None))
    cum = np.zeros_like(log_sin2)
    cum[:, 1:] = np.cumsum(log_sin2[:, :-1], axis=1)
    mean_log = cum.mean(axis=0)
    del cum, log_sin2
    sens = np.exp(mean_log - mean_log.max())
    return sens


def assign_tiers(sensitivity: np.ndarray, total_bits: int,
                 angle_ranges: np.ndarray) -> np.ndarray:
    M = len(sensitivity)
    ladder_bits = TIER_BITS_ARR

    def distortion(tier_bits: int, r: float) -> float:
        if tier_bits == 0:
            return (r * r) / 12.0
        if tier_bits <= 8:
            w = r / (1 << tier_bits)
            return (w * w) / 12.0
        if tier_bits == 16:
            return (r * (2.0 ** -10)) ** 2 / 12.0
        return (r * (2.0 ** -23)) ** 2 / 12.0

    tiers  = np.zeros(M, dtype=np.int32)
    c_dist = np.array([sensitivity[i] * distortion(0, angle_ranges[i])
                       for i in range(M)])
    used   = 0

    def step(i: int):
        cur = tiers[i]
        if cur >= len(ladder_bits) - 1:
            return None
        nxt = cur + 1
        extra = ladder_bits[nxt] - ladder_bits[cur]
        nd    = sensitivity[i] * distortion(ladder_bits[nxt], angle_ranges[i])
        gain  = c_dist[i] - nd
        return (gain / extra if extra > 0 else -np.inf, extra, nxt, nd)

    opts = [step(i) for i in range(M)]
    while used < total_bits:
        best_i, best_ratio, best_t = -1, -np.inf, None
        for i, o in enumerate(opts):
            if o is None:
                continue
            if o[1] > total_bits - used:
                continue
            if o[0] > best_ratio:
                best_ratio, best_i, best_t = o[0], i, o
        if best_i < 0 or best_ratio <= 0:
            break
        _, extra, new_tier, new_d = best_t
        tiers[best_i] = new_tier
        c_dist[best_i] = new_d
        used += extra
        opts[best_i] = step(best_i)
    return tiers


def apply_tiers(angles: np.ndarray, tiers: np.ndarray,
                angle_ranges: np.ndarray,
                clip_means: np.ndarray) -> np.ndarray:
    """Quantize then dequantize using per-angle tier assignment."""
    N, M = angles.shape
    out = np.empty((N, M), dtype=np.float32)
    out[:] = clip_means[None, :].astype(np.float32)
    for tier_id, bits in enumerate(TIER_BITS_ARR):
        idx = np.where(tiers == tier_id)[0]
        if not len(idx):
            continue
        if bits == 0:
            continue  # already filled with clip_means
        r_vec = angle_ranges[idx].astype(np.float32)
        if bits <= 8:
            levels = 1 << bits
            a = np.clip(angles[:, idx], 0.0, r_vec - 1e-6)
            q = np.clip(np.floor(a / r_vec * levels).astype(np.int32), 0, levels - 1)
            out[:, idx] = (q.astype(np.float32) + 0.5) * (r_vec / levels)
        elif bits == 16:
            out[:, idx] = angles[:, idx].astype(np.float16).astype(np.float32)
        else:
            out[:, idx] = angles[:, idx].astype(np.float32)
    return out


# -- Residual int8 (from geo3.py) --

def residual_int8_stack(original: np.ndarray, stage1: np.ndarray) -> np.ndarray:
    residual = original - stage1
    r_max = float(np.abs(residual).max())
    if r_max < 1e-12:
        return stage1.copy()
    levels = 127
    scale  = r_max / levels
    codes  = np.clip(np.round(residual / scale).astype(np.int32), -levels, levels).astype(np.int8)
    recon  = stage1 + codes.astype(np.float32) * scale
    norms  = np.linalg.norm(recon, axis=1, keepdims=True)
    return recon / np.clip(norms, 1e-12, None)


# ---------------------------------------------------------------------------
# Run all schemes
# ---------------------------------------------------------------------------

BIT_BUDGETS = [1536, 3072, 4608, 6144, 9216, 12288]


def run_all(
    c_vecs: np.ndarray,
    q_vecs: np.ndarray,
    c_ids: list[str],
    q_ids: list[str],
    qrels: dict[str, dict[str, int]],
    out_dir: Path,
) -> list[EvalResult]:
    dim     = c_vecs.shape[1]
    n_angles = dim - 1
    results: list[EvalResult] = []

    def ev(label, bits, cv, qv, family=""):
        print(f"  evaluating: {label} ...")
        r = evaluate(label, bits, qv, cv, q_ids, c_ids, qrels, family=family)
        print(f"    NDCG@10={r.ndcg10:.4f}  R@100={r.recall100:.4f}  {r.score_ms/1000:.1f}s")
        results.append(r)
        return r

    # ---- 1. Float32 baseline ----
    ev("float32 baseline", 32 * dim, c_vecs, q_vecs, family="baseline")

    # ---- 2. Cartesian int quantization ----
    for bits, label in [(8, "Cartesian int8"), (4, "Cartesian int4"), (2, "Cartesian int2")]:
        cv = cartesian_quant(c_vecs, bits)
        qv = cartesian_quant(q_vecs, bits)
        ev(label, bits * dim, cv, qv, family="cartesian")
        del cv, qv;  gc.collect()

    # ---- 3. Convert to angles (shared for all angle schemes) ----
    print("\nConverting corpus to hyperspherical angles (may take a few minutes) ...")
    t0 = time.perf_counter()
    c_angles = cartesian_to_hyperspherical(c_vecs)
    q_angles = cartesian_to_hyperspherical(q_vecs)
    gc.collect()
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    rt_err = float(np.mean(np.linalg.norm(
        recon_from_angles(c_angles[:500]) - c_vecs[:500], axis=1)))
    print(f"  round-trip L2 error (first 500): {rt_err:.2e}")

    emp_std      = c_angles.std(axis=0)
    angle_means  = c_angles.mean(axis=0)
    angle_ranges = np.full(n_angles, math.pi, dtype=np.float32)
    angle_ranges[-1] = 2 * math.pi
    sens = estimate_sensitivity(c_angles)
    gc.collect()

    # ---- 4. CLIPPING VALIDATION: 0-bit (all angles at corpus mean) ----
    print("\n--- Clipping validation ---")
    clip_cv = recon_from_angles(np.broadcast_to(angle_means, c_angles.shape).copy())
    clip_qv = recon_from_angles(np.broadcast_to(angle_means, q_angles.shape).copy())
    ev("CLIPPED (0-bit, all @ mean)", 0, clip_cv, clip_qv, family="clipping")
    del clip_cv, clip_qv;  gc.collect()

    half_angles_c = c_angles.copy()
    half_angles_c[:, n_angles // 2:] = angle_means[n_angles // 2:]
    half_angles_q = q_angles.copy()
    half_angles_q[:, n_angles // 2:] = angle_means[n_angles // 2:]
    hcv = recon_from_angles(half_angles_c);  del half_angles_c
    hqv = recon_from_angles(half_angles_q);  del half_angles_q
    ev("CLIPPED last 50% angles", 0, hcv, hqv, family="clipping")
    del hcv, hqv;  gc.collect()

    # ---- 5. Angle-uniform sweep ----
    print("\n--- Angle uniform quantization ---")
    for budget in BIT_BUDGETS:
        avg_bits = max(1, budget // n_angles)
        deq_c = angle_uniform_quant(c_angles, avg_bits)
        deq_q = angle_uniform_quant(q_angles, avg_bits)
        cv = recon_from_angles(deq_c);  del deq_c
        qv = recon_from_angles(deq_q);  del deq_q
        ev(f"angle-uniform {avg_bits}b/angle ({budget}b total)", budget, cv, qv, family="angle-uniform")
        del cv, qv;  gc.collect()

    # ---- 6. Jacobian-aware sweep (propagation prior) ----
    print("\n--- Angle Jacobian-prior quantization ---")
    for budget in BIT_BUDGETS:
        bits_arr = jacobian_bit_allocation(n_angles, budget)
        deq_c = angle_jacobian_quant(c_angles, bits_arr)
        deq_q = angle_jacobian_quant(q_angles, bits_arr)
        cv = recon_from_angles(deq_c);  del deq_c
        qv = recon_from_angles(deq_q);  del deq_q
        ev(f"angle-jacobian {budget}b (min={bits_arr.min()} med={int(np.median(bits_arr))} max={bits_arr.max()})",
           int(bits_arr.sum()), cv, qv, family="angle-jacobian")
        del cv, qv;  gc.collect()

    # ---- 7. Empirical-std bit allocation ----
    print("\n--- Angle empirical-std quantization ---")
    for budget in BIT_BUDGETS:
        bits_arr = jacobian_bit_allocation(n_angles, budget, empirical_std=emp_std)
        deq_c = angle_jacobian_quant(c_angles, bits_arr)
        deq_q = angle_jacobian_quant(q_angles, bits_arr)
        cv = recon_from_angles(deq_c);  del deq_c
        qv = recon_from_angles(deq_q);  del deq_q
        ev(f"angle-emp-std {budget}b", int(bits_arr.sum()), cv, qv, family="angle-emp-std")
        del cv, qv;  gc.collect()

    # ---- 8. Dynamic tier ----
    print("\n--- Dynamic tier quantization ---")
    for budget in BIT_BUDGETS:
        tiers_arr   = assign_tiers(sens, budget, angle_ranges)
        actual_bits = int(sum(TIER_BITS_ARR[t] for t in tiers_arr))
        deq_c = apply_tiers(c_angles, tiers_arr, angle_ranges, angle_means)
        deq_q = apply_tiers(q_angles, tiers_arr, angle_ranges, angle_means)
        cv = recon_from_angles(deq_c);  del deq_c
        qv = recon_from_angles(deq_q);  del deq_q
        ev(f"dyn-tier {budget}b (actual={actual_bits}b)", actual_bits, cv, qv, family="dyn-tier")

        # ---- 9. Residual int8 on top of dynamic tier ----
        cv_stack = residual_int8_stack(c_vecs, cv)
        qv_stack = residual_int8_stack(q_vecs, qv)
        del cv, qv;  gc.collect()
        ev(f"dyn-tier {budget}b + residual-int8",
           actual_bits + 8 * dim, cv_stack, qv_stack, family="dyn-tier+residual")
        del cv_stack, qv_stack;  gc.collect()

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_table(results: list[EvalResult], path: Path) -> None:
    float32_bits = next((r.bits_per_vec for r in results if "float32" in r.label), 1)
    header = f"{'method':<60} {'bits/vec':>10} {'ratio':>7} {'NDCG@10':>9} {'R@100':>8} {'score_s':>9}"
    sep    = "-" * len(header)
    lines  = [header, sep]
    for r in results:
        ratio = float32_bits / r.bits_per_vec if r.bits_per_vec > 0 else float("inf")
        lines.append(
            f"{r.label:<60} {r.bits_per_vec:>10d} {ratio:>7.1f}x "
            f"{r.ndcg10:>9.4f} {r.recall100:>8.4f} {r.score_ms/1000:>9.2f}"
        )
    text = "\n".join(lines)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"\ntable written to: {path}")


def write_csv(results: list[EvalResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "family", "bits_per_vec", "compression_ratio",
                    "ndcg10", "recall100", "score_ms"])
        float32_bits = next((r.bits_per_vec for r in results if "float32" in r.label), 1)
        for r in results:
            ratio = float32_bits / r.bits_per_vec if r.bits_per_vec > 0 else float("inf")
            w.writerow([r.label, r.family, r.bits_per_vec, f"{ratio:.3f}",
                        f"{r.ndcg10:.4f}", f"{r.recall100:.4f}", f"{r.score_ms:.1f}"])
    print(f"CSV  written to: {path}")


def plot_tradeoff(results: list[EvalResult], path: Path) -> None:
    float32_bits = next((r.bits_per_vec for r in results if "float32" in r.label), 1)
    baseline_ndcg = next((r.ndcg10 for r in results if "float32" in r.label), 1.0)

    families = {
        "cartesian":      ("Cartesian int (8/4/2-bit)",   "o",  "#e74c3c"),
        "angle-uniform":  ("Angle uniform",                "s",  "#3498db"),
        "angle-jacobian": ("Angle Jacobian-prior",         "^",  "#2ecc71"),
        "angle-emp-std":  ("Angle empirical-std",          "D",  "#9b59b6"),
        "dyn-tier":       ("Dynamic tier",                 "P",  "#f39c12"),
        "dyn-tier+residual": ("Dyn-tier + residual int8",  "X",  "#1abc9c"),
        "clipping":       ("Clipping experiments",         "v",  "#95a5a6"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes

    for fam, (name, marker, color) in families.items():
        pts = [(float32_bits / r.bits_per_vec, r.ndcg10)
               for r in results
               if r.family == fam and r.bits_per_vec > 0]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker=marker, label=name, color=color,
                 linewidth=1.5, markersize=7)
        ax2.plot(xs, ys, marker=marker, label=name, color=color,
                 linewidth=1.5, markersize=7)

    for ax in (ax1, ax2):
        ax.axhline(baseline_ndcg, color="black", linestyle="--",
                   linewidth=1.2, label="float32 baseline")
        ax.set_xlabel("Compression ratio (vs float32)", fontsize=12)
        ax.set_ylabel("NDCG@10", fontsize=12)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    ax1.set_title("Compression vs NDCG@10 (linear scale)", fontsize=13)
    ax2.set_title("Compression vs NDCG@10 (log x scale)", fontsize=13)
    ax2.set_xscale("log")

    fig.suptitle("Quantization scheme comparison on FEVER (BEIR)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", required=True,
                    help="Root of embedding output dir for one dataset "
                         "(e.g. embeddings/fever). Must contain corpus/ and queries/ sub-dirs.")
    ap.add_argument("--data-dir",  required=True,
                    help="BEIR dataset directory (e.g. bier-data/fever). "
                         "Used to load qrels.")
    ap.add_argument("--embed-tag", default="",
                    help="Chunk filename tag (e.g. 'minilm-l6-vllm'). "
                         "Empty string means the plain OAI layout (no tag).")
    ap.add_argument("--max-corpus",  type=int, default=50_000,
                    help="Max corpus docs to load (default 50k, safe for t3.medium 4GB). "
                         "Use 0 for no limit (requires ~33 GB for full FEVER).")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="Max queries to evaluate (0 = all).")
    ap.add_argument("--output-dir",  default="eval_results",
                    help="Where to write table, CSV, and plot.")
    ap.add_argument("--batch-q", type=int, default=256,
                    help="Query batch size for scoring (reduce if OOM).")
    args = ap.parse_args()

    embed_dir = Path(args.embed_dir).expanduser().resolve()
    data_dir  = Path(args.data_dir).expanduser().resolve()
    out_dir   = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_corpus = args.max_corpus if args.max_corpus > 0 else None

    print(f"Loading corpus embeddings from {embed_dir / 'corpus'} ...")
    c_vecs, c_ids = load_split(embed_dir / "corpus", args.embed_tag, max_corpus)
    print(f"  loaded {c_vecs.shape[0]:,} corpus vectors  dim={c_vecs.shape[1]}")

    print(f"Loading query embeddings from {embed_dir / 'queries'} ...")
    q_max = args.max_queries if args.max_queries > 0 else None
    q_vecs, q_ids = load_split(embed_dir / "queries", args.embed_tag, q_max)
    print(f"  loaded {q_vecs.shape[0]:,} query vectors")

    print(f"Loading qrels from {data_dir} ...")
    all_qrels = load_qrels(data_dir)

    # keep only queries whose relevant docs are in our corpus sample
    c_id_set = set(c_ids)
    filtered_qids = [qid for qid in q_ids
                     if qid in all_qrels and
                     any(cid in c_id_set for cid in all_qrels[qid])]
    print(f"  {len(filtered_qids):,} / {len(q_ids):,} queries have relevant docs in corpus sample")

    if not filtered_qids:
        print("ERROR: no queries have relevant docs in the loaded corpus sample. "
              "Increase --max-corpus or check --embed-tag.")
        return

    # Reorder q_vecs to match filtered_qids
    qid_to_idx = {qid: i for i, qid in enumerate(q_ids)}
    filt_idx   = [qid_to_idx[qid] for qid in filtered_qids]
    q_vecs_f   = q_vecs[filt_idx]
    qrels_f    = {qid: all_qrels[qid] for qid in filtered_qids}

    print(f"\nRunning evaluation on {len(filtered_qids):,} queries x {len(c_ids):,} corpus docs")
    print(f"Dim={c_vecs.shape[1]}  float32 baseline = {32 * c_vecs.shape[1]:,} bits/vec\n")

    results = run_all(c_vecs, q_vecs_f, c_ids, filtered_qids, qrels_f, out_dir)

    write_table(results, out_dir / "results_table.txt")
    write_csv(results,   out_dir / "results.csv")
    plot_tradeoff(results, out_dir / "tradeoff.png")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
