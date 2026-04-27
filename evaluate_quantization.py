"""
Quantization evaluation on real BEIR embeddings.

Full pipeline under test:
    embedding (float32)
        -> hyperspherical angles (θ₁ … θ_{dim-1})
        -> rank angles by Jacobian sensitivity
        -> assign precision per angle (greedy tier OR uniform OR truncation/clipping)
        -> optional: residual int8 stacking on top
        -> cosine similarity retrieval -> NDCG@10 / Recall@100

Experiment groups (all at matched bit budgets for fair comparison):
  A. Baselines:      float32, Cartesian int8/int4/int2
  B. Angle uniform:  angle->int8/int4/int2 (no ranking, no clipping)
  C. Clipping:       positional clip vs Jacobian clip at 0%→100% thresholds
  D. Greedy tier:    dynamic per-angle precision (CLIPPED/int2/int4/int8/f16/f32)
                     assigned by greedy knapsack under bit budgets
  E. Residual:       greedy tier + per-vector-scaled int8 residual stacking

Ablation table:  at a fixed bit budget, measures marginal benefit of each component:
    Cartesian int -> +Angles -> +Jacobian rank -> +Greedy tier -> +Residual

Outputs (in --output-dir):
  results_table.txt      all schemes
  ablation_table.txt     per-component marginal benefit at fixed budget
  tradeoff_table.txt     per clipping threshold: NDCG, bits, MB saved
  results.csv
  plot_main.png          Cartesian vs Angle (groups A & B)
  plot_clipping.png      positional vs Jacobian clipping (group C)
  plot_greedy.png        greedy tier sweep vs clipping at equal budgets (D vs C)
  plot_tradeoff.png      all groups, compression ratio vs NDCG
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------

def _glob_chunks(split_dir: Path, tag: str) -> list[Path]:
    if tag:
        return sorted(split_dir.glob(f"chunk_*.{tag}.npy"))
    return sorted(p for p in split_dir.glob("chunk_*.npy")
                  if p.name.count(".") == 1)


def load_split(split_dir: Path, tag: str,
               max_vecs: int | None = None) -> tuple[np.ndarray, list[str]]:
    chunks = _glob_chunks(split_dir, tag)
    if not chunks:
        raise FileNotFoundError(
            f"No chunks in {split_dir} with tag='{tag}'. "
            "Check --embed-tag and that embeddings exist.")
    all_vecs: list[np.ndarray] = []
    all_ids:  list[str] = []
    for p in chunks:
        arr   = np.load(p)
        stem  = p.name[:-len(".npy")]
        ids_p = split_dir / (stem + ".ids.txt")
        with ids_p.open("r", encoding="utf-8") as f:
            ids = [ln.rstrip("\n") for ln in f]
        assert arr.shape[0] == len(ids), f"shape mismatch {p.name}"
        all_vecs.append(arr)
        all_ids.extend(ids)
        if max_vecs and sum(v.shape[0] for v in all_vecs) >= max_vecs:
            break
    vecs = np.concatenate(all_vecs, axis=0).astype(np.float32)
    if max_vecs:
        vecs, all_ids = vecs[:max_vecs], all_ids[:max_vecs]
    return vecs, all_ids


# ---------------------------------------------------------------------------
# Qrels
# ---------------------------------------------------------------------------

def load_qrels(data_dir: Path) -> dict[str, dict[str, int]]:
    for name in ("test.tsv", "dev.tsv", "train.tsv"):
        p = data_dir / "qrels" / name
        if p.exists():
            print(f"loading qrels from {p}")
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
# Metrics
# ---------------------------------------------------------------------------

def dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(top_ids: np.ndarray, relevant: dict[str, int], k: int) -> float:
    rels  = [relevant.get(cid, 0) for cid in top_ids[:k]]
    ideal = sorted(relevant.values(), reverse=True)
    idcg  = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg else 0.0


def recall_at_k(top_ids: np.ndarray, relevant: dict[str, int], k: int) -> float:
    return sum(1 for cid in top_ids[:k] if cid in relevant) / len(relevant) \
           if relevant else 0.0


@dataclass
class EvalResult:
    label:       str
    group:       str
    bits_per_vec: int
    ndcg10:      float
    recall100:   float
    score_ms:    float


def evaluate(
    label: str, group: str, bits_per_vec: int,
    q_vecs: np.ndarray, c_vecs: np.ndarray,
    q_ids: list[str], c_ids: list[str],
    qrels: dict[str, dict[str, int]],
    batch_q: int = 256,
) -> EvalResult:
    c_arr = np.array(c_ids)
    top_k = 100
    ndcg_sc: list[float] = []
    rec_sc:  list[float] = []
    t0 = time.perf_counter()
    for qi in range(0, len(q_ids), batch_q):
        qb   = q_vecs[qi: qi + batch_q]
        sims = qb @ c_vecs.T
        tidx = np.argpartition(-sims, kth=min(top_k, sims.shape[1] - 1),
                               axis=1)[:, :top_k]
        for bi in range(qb.shape[0]):
            qid = q_ids[qi + bi]
            rel = qrels.get(qid)
            if not rel:
                continue
            rs   = sims[bi, tidx[bi]]
            ord_ = np.argsort(-rs)
            ranked = c_arr[tidx[bi][ord_]]
            ndcg_sc.append(ndcg_at_k(ranked, rel, 10))
            rec_sc.append(recall_at_k(ranked, rel, 100))
    elapsed = time.perf_counter() - t0
    return EvalResult(label=label, group=group, bits_per_vec=bits_per_vec,
                      ndcg10=float(np.mean(ndcg_sc))  if ndcg_sc else 0.0,
                      recall100=float(np.mean(rec_sc)) if rec_sc  else 0.0,
                      score_ms=elapsed * 1000)


# ---------------------------------------------------------------------------
# Coordinate conversion (float32 throughout)
# ---------------------------------------------------------------------------

def to_angles(x: np.ndarray) -> np.ndarray:
    x   = x.astype(np.float32, copy=False)
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
    a = a.astype(np.float32, copy=False)
    N, m = a.shape
    x    = np.zeros((N, m + 1), dtype=np.float32)
    sp   = np.ones(N, dtype=np.float32)
    for i in range(m):
        x[:, i] = sp * np.cos(a[:, i])
        sp = sp * np.sin(a[:, i])
    x[:, m] = sp
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-7, None)


# ---------------------------------------------------------------------------
# Jacobian sensitivity & ranking
# ---------------------------------------------------------------------------

def jacobian_sensitivity(angles_calib: np.ndarray) -> np.ndarray:
    """
    Per-angle sensitivity = E[sin²(θ₁)·...·sin²(θᵢ₋₁)] under the calibration set.
    Early angles have high sensitivity (not yet attenuated by sin-products).
    Returns (M,) array; higher = more important to keep.
    """
    N, M     = angles_calib.shape
    log_sin2 = 2.0 * np.log(np.clip(np.abs(np.sin(angles_calib)), 1e-7, None))
    cum      = np.zeros((N, M), dtype=np.float32)
    cum[:, 1:] = np.cumsum(log_sin2[:, :-1], axis=1)
    mean_log   = cum.mean(axis=0)
    del cum, log_sin2
    sens = np.exp(mean_log - mean_log.max())
    return sens


def rank_by_sensitivity(sens: np.ndarray) -> np.ndarray:
    """Return indices sorted most → least sensitive."""
    return np.argsort(-sens)


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def quant_cartesian(vecs: np.ndarray, bits: int) -> np.ndarray:
    levels = 1 << bits
    scale  = (levels - 1) / 2.0
    q   = np.clip(np.round(np.clip(vecs, -1.0, 1.0) * scale + scale / 2),
                  0, levels - 1).astype(np.int32)
    out = (q.astype(np.float32) + 0.5) / scale - 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-7, None)


def _uquant(angles: np.ndarray, bits: int, hi: np.ndarray) -> np.ndarray:
    """Uniform scalar quantization of a subset of angles."""
    levels = 1 << bits
    a = np.clip(angles, 0.0, hi - 1e-6)
    q = np.clip(np.floor(a / hi * levels).astype(np.int32), 0, levels - 1)
    return (q.astype(np.float32) + 0.5) * (hi / levels)


def quant_angles_uniform(angles: np.ndarray, bits: int, hi: np.ndarray) -> np.ndarray:
    return _uquant(angles, bits, hi)


def quant_angles_clip(angles: np.ndarray, hi: np.ndarray,
                      keep_bits: int, keep_n: int,
                      clip_means: np.ndarray,
                      keep_idx: np.ndarray | None = None) -> np.ndarray:
    """
    keep_n angles at keep_bits; rest replaced by corpus mean (0-bit).
    keep_idx: Jacobian ranking (None = positional).
    """
    out = clip_means[None, :].repeat(angles.shape[0], axis=0).astype(np.float32)
    if keep_n <= 0:
        return out
    idx = keep_idx[:keep_n] if keep_idx is not None else np.arange(keep_n)
    out[:, idx] = _uquant(angles[:, idx], keep_bits, hi[idx])
    return out


# ---------------------------------------------------------------------------
# Greedy tier assignment
# ---------------------------------------------------------------------------

# Tier IDs and their bit costs
TIER_BITS = [0, 2, 4, 8, 16, 32]   # CLIPPED, INT2, INT4, INT8, FP16, FP32


def _distortion(tier_bits: int, r: float) -> float:
    if tier_bits == 0:
        return (r * r) / 12.0
    if tier_bits <= 8:
        w = r / (1 << tier_bits)
        return (w * w) / 12.0
    if tier_bits == 16:
        return (r * 2 ** -10) ** 2 / 12.0
    return (r * 2 ** -23) ** 2 / 12.0


def greedy_tier_assign(sens: np.ndarray, total_bits: int,
                       angle_ranges: np.ndarray) -> np.ndarray:
    """
    Assign each angle to a precision tier under a total bit budget.
    Greedy: start everyone at CLIPPED (0 bits), upgrade whichever angle gives
    the best distortion-reduction-per-extra-bit ratio until budget exhausted.

    Returns: (M,) int array of tier indices (index into TIER_BITS).
    """
    M      = len(sens)
    tiers  = np.zeros(M, dtype=np.int32)   # all start at CLIPPED
    c_dist = np.array([sens[i] * _distortion(0, float(angle_ranges[i]))
                       for i in range(M)], dtype=np.float64)
    used   = 0

    def next_step(i: int):
        cur = tiers[i]
        if cur >= len(TIER_BITS) - 1:
            return None
        nxt        = cur + 1
        extra_bits = TIER_BITS[nxt] - TIER_BITS[cur]
        new_d      = sens[i] * _distortion(TIER_BITS[nxt], float(angle_ranges[i]))
        gain       = c_dist[i] - new_d
        return (gain / extra_bits if extra_bits > 0 else -np.inf,
                extra_bits, nxt, new_d)

    opts = [next_step(i) for i in range(M)]

    while used < total_bits:
        best_i, best_r, best_t = -1, -np.inf, None
        for i, o in enumerate(opts):
            if o is None or o[1] > total_bits - used:
                continue
            if o[0] > best_r:
                best_r, best_i, best_t = o[0], i, o
        if best_i < 0 or best_r <= 0:
            break
        _, extra, new_tier, new_d = best_t
        tiers[best_i]  = new_tier
        c_dist[best_i] = new_d
        used += extra
        opts[best_i] = next_step(best_i)

    return tiers


def apply_greedy_tiers(angles: np.ndarray, tiers: np.ndarray,
                       hi: np.ndarray, clip_means: np.ndarray) -> np.ndarray:
    """Quantize+dequantize angles using per-angle tier assignment."""
    N, M = angles.shape
    out  = clip_means[None, :].repeat(N, axis=0).astype(np.float32)
    for tier_id, bits in enumerate(TIER_BITS):
        idx = np.where(tiers == tier_id)[0]
        if not len(idx) or bits == 0:
            continue
        r_vec = hi[idx]
        if bits <= 8:
            out[:, idx] = _uquant(angles[:, idx], bits, r_vec)
        elif bits == 16:
            out[:, idx] = angles[:, idx].astype(np.float16).astype(np.float32)
        else:
            out[:, idx] = angles[:, idx]
    return out


def tier_bits_used(tiers: np.ndarray) -> int:
    return int(sum(TIER_BITS[t] for t in tiers))


# ---------------------------------------------------------------------------
# Residual int8 (per-vector scaling -- fixed version)
# ---------------------------------------------------------------------------

def residual_int8_pv(original: np.ndarray, stage1: np.ndarray) -> np.ndarray:
    """
    Quantize the residual (original - stage1) with per-vector int8 scaling.
    Per-vector scale means each vector's dynamic range is used independently,
    so one outlier vector doesn't crush resolution for everyone else.

    Storage cost: 8 * dim additional bits (for the int8 codes).
    The per-vector scale factor (1 float32 per vector) is negligible overhead.
    """
    residual = original - stage1                             # (N, dim)
    r_max    = np.abs(residual).max(axis=1, keepdims=True)  # (N, 1) per-vector
    r_max    = np.clip(r_max, 1e-12, None)
    scale    = r_max / 127.0
    codes    = np.clip(np.round(residual / scale).astype(np.int32),
                       -127, 127).astype(np.int8)
    recon    = stage1 + codes.astype(np.float32) * scale
    norms    = np.linalg.norm(recon, axis=1, keepdims=True)
    return recon / np.clip(norms, 1e-12, None)


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

# Bit budgets for greedy tier sweep (float32 = 32*dim bits)
TIER_BUDGETS  = [1536, 3072, 4608, 6144, 9216, 12288]
CLIP_FRACS    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TRUNC_FRACS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Fixed budget for ablation table (~8x compression for 1536-dim OAI embeddings)
ABLATION_BITS = 6144


def run_all(
    c_vecs: np.ndarray, q_vecs: np.ndarray,
    c_ids: list[str], q_ids: list[str],
    qrels: dict[str, dict[str, int]],
    batch_q: int,
) -> tuple[list[EvalResult], list[EvalResult], list[EvalResult],
           list[EvalResult], list[EvalResult]]:
    """
    Returns: results, clip_pos, clip_jac, greedy_sweep, ablation
    """
    dim      = c_vecs.shape[1]
    n_angles = dim - 1
    hi       = np.full(n_angles, math.pi, dtype=np.float32)
    hi[-1]   = 2 * math.pi

    results:      list[EvalResult] = []
    clip_pos:     list[EvalResult] = []
    clip_jac:     list[EvalResult] = []
    greedy_sweep: list[EvalResult] = []
    ablation:     list[EvalResult] = []

    def ev(label, group, bits, cv, qv, store=results):
        print(f"  {label}")
        r = evaluate(label, group, bits, qv, cv, q_ids, c_ids, qrels, batch_q)
        print(f"    NDCG@10={r.ndcg10:.4f}  R@100={r.recall100:.4f}  "
              f"{r.score_ms/1000:.1f}s")
        store.append(r)
        return r

    # -------------------------------------------------------------------
    # A. Baselines
    # -------------------------------------------------------------------
    print("\n=== A. Baselines ===")
    r_f32 = ev("float32 baseline", "baseline", 32 * dim, c_vecs, q_vecs)

    for bits in (8, 4, 2):
        cv = quant_cartesian(c_vecs, bits)
        qv = quant_cartesian(q_vecs, bits)
        ev(f"Cartesian int{bits}", "cartesian", bits * dim, cv, qv)
        del cv, qv; gc.collect()

    # -------------------------------------------------------------------
    # Convert to angles + Jacobian ranking (shared for all remaining groups)
    # -------------------------------------------------------------------
    print("\n  [converting to angles + computing Jacobian ranking]")
    c_angles   = to_angles(c_vecs)
    q_angles   = to_angles(q_vecs)
    clip_means = c_angles.mean(axis=0)
    sens       = jacobian_sensitivity(c_angles)
    jac_rank   = rank_by_sensitivity(sens)
    angle_ranges = hi.copy()
    gc.collect()

    print(f"  sensitivity: max={sens[jac_rank[0]]:.4f}  "
          f"min={sens[jac_rank[-1]]:.2e}  "
          f"ratio={sens[jac_rank[0]]/max(sens[jac_rank[-1]], 1e-12):.0f}x")

    # -------------------------------------------------------------------
    # B. Angle uniform (no ranking, no clipping)
    # -------------------------------------------------------------------
    print("\n=== B. Angle uniform quantization ===")
    for bits in (8, 4, 2):
        deq_c = quant_angles_uniform(c_angles, bits, hi)
        deq_q = quant_angles_uniform(q_angles, bits, hi)
        cv = from_angles(deq_c); del deq_c
        qv = from_angles(deq_q); del deq_q
        ev(f"Angle int{bits} uniform", "angle-uniform", bits * n_angles, cv, qv)
        del cv, qv; gc.collect()

    # -------------------------------------------------------------------
    # C. Clipping sweep: positional vs Jacobian
    # -------------------------------------------------------------------
    print("\n=== C. Clipping sweep (positional vs Jacobian) ===")
    for frac in CLIP_FRACS:
        keep_n = int(round(n_angles * (1.0 - frac)))
        bits   = keep_n * 8

        # Positional (naive)
        deq_c = quant_angles_clip(c_angles, hi, 8, keep_n, clip_means)
        deq_q = quant_angles_clip(q_angles, hi, 8, keep_n, clip_means)
        cv = from_angles(deq_c); del deq_c
        qv = from_angles(deq_q); del deq_q
        label = f"Pos-clip {int(frac*100):3d}% (keep {keep_n}@int8={bits}b)"
        r = ev(label, "clip-positional", bits, cv, qv)
        clip_pos.append(r)
        del cv, qv; gc.collect()

        # Jacobian-ranked
        deq_c = quant_angles_clip(c_angles, hi, 8, keep_n, clip_means, jac_rank)
        deq_q = quant_angles_clip(q_angles, hi, 8, keep_n, clip_means, jac_rank)
        cv = from_angles(deq_c); del deq_c
        qv = from_angles(deq_q); del deq_q
        label = f"Jac-clip {int(frac*100):3d}% (keep {keep_n}@int8={bits}b)"
        r = ev(label, "clip-jacobian", bits, cv, qv)
        clip_jac.append(r)
        del cv, qv; gc.collect()

    # -------------------------------------------------------------------
    # D. Greedy tier sweep (Jacobian-informed, mixed precision)
    # -------------------------------------------------------------------
    print("\n=== D. Greedy tier sweep ===")
    for budget in TIER_BUDGETS:
        tiers      = greedy_tier_assign(sens, budget, angle_ranges)
        actual     = tier_bits_used(tiers)
        tier_counts = {b: int(np.sum(tiers == i))
                       for i, b in enumerate(TIER_BITS)}
        print(f"  budget={budget}b -> actual={actual}b  "
              f"clip={tier_counts[0]} i2={tier_counts[2]} i4={tier_counts[4]} "
              f"i8={tier_counts[8]} f16={tier_counts[16]} f32={tier_counts[32]}")

        deq_c = apply_greedy_tiers(c_angles, tiers, hi, clip_means)
        deq_q = apply_greedy_tiers(q_angles, tiers, hi, clip_means)
        cv = from_angles(deq_c); del deq_c
        qv = from_angles(deq_q); del deq_q
        label = f"Greedy-tier budget={budget}b (actual={actual}b)"
        r = ev(label, "greedy-tier", actual, cv, qv)
        greedy_sweep.append(r)

        # Greedy + residual int8
        cv_res = residual_int8_pv(c_vecs, cv)
        qv_res = residual_int8_pv(q_vecs, qv)
        del cv, qv; gc.collect()
        label_res = f"Greedy-tier {budget}b + residual-int8"
        r_res = ev(label_res, "greedy+residual", actual + 8 * dim, cv_res, qv_res)
        del cv_res, qv_res; gc.collect()

    # -------------------------------------------------------------------
    # E. Truncation tradeoff: int8/int4/int2 (Jacobian-ranked)
    # Shows whether finer precision on kept angles is worth it.
    # -------------------------------------------------------------------
    print("\n=== E. Truncation tradeoff (Jacobian-ranked, int8/int4/int2) ===")
    for keep_bits in (8, 4, 2):
        for kf in TRUNC_FRACS:
            keep_n = int(round(n_angles * kf))
            bits   = keep_n * keep_bits
            deq_c  = quant_angles_clip(c_angles, hi, keep_bits, keep_n,
                                       clip_means, jac_rank)
            deq_q  = quant_angles_clip(q_angles, hi, keep_bits, keep_n,
                                       clip_means, jac_rank)
            cv = from_angles(deq_c); del deq_c
            qv = from_angles(deq_q); del deq_q
            label = f"Jac-trunc int{keep_bits} keep={int(kf*100)}%"
            ev(label, f"jac-trunc-int{keep_bits}", bits, cv, qv)
            del cv, qv; gc.collect()

    # -------------------------------------------------------------------
    # F. Ablation table (fixed budget = ABLATION_BITS)
    # Each row adds one component; measures marginal NDCG gain.
    # -------------------------------------------------------------------
    print(f"\n=== F. Ablation at {ABLATION_BITS}b fixed budget ===")
    keep_n_ab = ABLATION_BITS // 8                        # for int8 clipping
    keep_n_i4 = ABLATION_BITS // 4                        # for int4 clipping

    # F1: Cartesian int8 at same budget (need to pick equivalent bits/dim)
    # ABLATION_BITS bits on Cartesian: bits_per_dim = ABLATION_BITS // dim
    cart_bits = max(1, ABLATION_BITS // dim)
    cv = quant_cartesian(c_vecs, cart_bits)
    qv = quant_cartesian(q_vecs, cart_bits)
    r  = ev(f"[Ablation] Cartesian int{cart_bits} (~{ABLATION_BITS}b)",
            "ablation", cart_bits * dim, cv, qv, store=ablation)
    del cv, qv; gc.collect()

    # F2: Angle int8 uniform (no ranking, no clipping)
    deq_c = quant_angles_uniform(c_angles, 8, hi)
    deq_q = quant_angles_uniform(q_angles, 8, hi)
    cv = from_angles(deq_c); del deq_c
    qv = from_angles(deq_q); del deq_q
    ev(f"[Ablation] + Angles int8 uniform ({n_angles*8}b)",
       "ablation", n_angles * 8, cv, qv, store=ablation)
    # keep cv/qv for residual below
    del cv, qv; gc.collect()

    # F3: Angles + positional clip (no Jacobian ranking)
    deq_c = quant_angles_clip(c_angles, hi, 8, keep_n_ab, clip_means)
    deq_q = quant_angles_clip(q_angles, hi, 8, keep_n_ab, clip_means)
    cv = from_angles(deq_c); del deq_c
    qv = from_angles(deq_q); del deq_q
    ev(f"[Ablation] + Positional clip to {ABLATION_BITS}b",
       "ablation", keep_n_ab * 8, cv, qv, store=ablation)
    del cv, qv; gc.collect()

    # F4: Angles + Jacobian clip (ranking added)
    deq_c = quant_angles_clip(c_angles, hi, 8, keep_n_ab, clip_means, jac_rank)
    deq_q = quant_angles_clip(q_angles, hi, 8, keep_n_ab, clip_means, jac_rank)
    cv = from_angles(deq_c); del deq_c
    qv = from_angles(deq_q); del deq_q
    ev(f"[Ablation] + Jacobian clip to {ABLATION_BITS}b",
       "ablation", keep_n_ab * 8, cv, qv, store=ablation)
    del cv, qv; gc.collect()

    # F5: Greedy tier at ABLATION_BITS budget
    tiers  = greedy_tier_assign(sens, ABLATION_BITS, angle_ranges)
    actual = tier_bits_used(tiers)
    deq_c  = apply_greedy_tiers(c_angles, tiers, hi, clip_means)
    deq_q  = apply_greedy_tiers(q_angles, tiers, hi, clip_means)
    cv = from_angles(deq_c); del deq_c
    qv = from_angles(deq_q); del deq_q
    ev(f"[Ablation] + Greedy tier ~{ABLATION_BITS}b (actual={actual}b)",
       "ablation", actual, cv, qv, store=ablation)

    # F6: Greedy + residual int8
    cv_res = residual_int8_pv(c_vecs, cv)
    qv_res = residual_int8_pv(q_vecs, qv)
    del cv, qv; gc.collect()
    ev(f"[Ablation] + Residual int8 ({actual + 8*dim}b total)",
       "ablation", actual + 8 * dim, cv_res, qv_res, store=ablation)
    del cv_res, qv_res; gc.collect()

    del c_angles, q_angles; gc.collect()
    return results, clip_pos, clip_jac, greedy_sweep, ablation


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_table(results: list[EvalResult], path: Path, title: str = "") -> None:
    base_bits = next((r.bits_per_vec for r in results if "float32" in r.label), 1)
    hdr = (f"{'method':<70} {'bits':>8} {'ratio':>7} "
           f"{'NDCG@10':>9} {'R@100':>8} {'s':>7}")
    sep = "-" * len(hdr)
    lines = ([title, "=" * len(hdr)] if title else []) + [hdr, sep]
    for r in results:
        ratio = base_bits / r.bits_per_vec if r.bits_per_vec > 0 else float("inf")
        lines.append(
            f"{r.label:<70} {r.bits_per_vec:>8d} {ratio:>7.1f}x "
            f"{r.ndcg10:>9.4f} {r.recall100:>8.4f} {r.score_ms/1000:>7.2f}")
    text = "\n".join(lines)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"table -> {path}")


def write_ablation_table(ablation: list[EvalResult], base_ndcg: float,
                         path: Path) -> None:
    hdr = (f"{'component':<55} {'bits':>8} {'ratio':>7} "
           f"{'NDCG@10':>9} {'ΔNDCG':>8} {'R@100':>8}")
    sep = "-" * len(hdr)
    lines = [f"Ablation table (fixed budget ~{ABLATION_BITS}b)",
             "=" * len(hdr), hdr, sep]
    prev_ndcg = base_ndcg
    base_bits = ablation[0].bits_per_vec if ablation else 1
    for r in ablation:
        ratio  = base_bits / r.bits_per_vec if r.bits_per_vec > 0 else float("inf")
        delta  = r.ndcg10 - prev_ndcg
        lines.append(
            f"{r.label:<55} {r.bits_per_vec:>8d} {ratio:>7.1f}x "
            f"{r.ndcg10:>9.4f} {delta:>+8.4f} {r.recall100:>8.4f}")
        prev_ndcg = r.ndcg10
    text = "\n".join(lines)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"ablation -> {path}")


def write_tradeoff_table(clip_pos: list[EvalResult], clip_jac: list[EvalResult],
                         base_ndcg: float, full_n: int, dim: int,
                         path: Path) -> None:
    f32_bytes = 32 * dim / 8
    hdr = (f"{'clip%':>6} {'bits':>7} {'ratio':>7} "
           f"{'NDCG-pos':>10} {'NDCG-jac':>10} {'Δ(jac-pos)':>12} "
           f"{'MB saved/5.4M':>15}")
    sep = "-" * len(hdr)
    lines = ["Tradeoff: clipping threshold vs quality vs storage",
             "=" * len(hdr), hdr, sep]
    for rp, rj in zip(clip_pos, clip_jac):
        m    = re.search(r"(\d+)%", rj.label)
        frac = int(m.group(1)) / 100.0 if m else 0.0
        ratio  = 32 * dim / rj.bits_per_vec if rj.bits_per_vec > 0 else float("inf")
        mb_saved = (f32_bytes - rj.bits_per_vec / 8) * full_n / 1e6
        delta  = rj.ndcg10 - rp.ndcg10
        lines.append(
            f"{frac*100:>5.0f}% {rj.bits_per_vec:>7d} {ratio:>7.1f}x "
            f"{rp.ndcg10:>10.4f} {rj.ndcg10:>10.4f} {delta:>+12.4f} "
            f"{mb_saved:>15,.0f}")
    text = "\n".join(lines)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"tradeoff table -> {path}")


def write_csv(results: list[EvalResult], path: Path) -> None:
    base_bits = next((r.bits_per_vec for r in results if "float32" in r.label), 1)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "group", "bits_per_vec", "compression_ratio",
                    "ndcg10", "recall100", "score_ms"])
        for r in results:
            ratio = base_bits / r.bits_per_vec if r.bits_per_vec > 0 else float("inf")
            w.writerow([r.label, r.group, r.bits_per_vec, f"{ratio:.3f}",
                        f"{r.ndcg10:.4f}", f"{r.recall100:.4f}", f"{r.score_ms:.1f}"])
    print(f"CSV -> {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ratio(r: EvalResult, base_bits: int) -> float:
    return base_bits / r.bits_per_vec if r.bits_per_vec > 0 else float("inf")


def plot_main(results: list[EvalResult], path: Path) -> None:
    base      = next(r for r in results if "float32" in r.label)
    base_bits = base.bits_per_vec
    cart      = [r for r in results if r.group == "cartesian"]
    angle     = [r for r in results if r.group == "angle-uniform"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([_ratio(r, base_bits) for r in cart],
               [r.ndcg10 for r in cart],
               color="#e74c3c", s=120, zorder=5, label="Cartesian int")
    for r in cart:
        b = r.bits_per_vec // dim if (dim := base_bits // 32) else 1
        ax.annotate(f"int{b}", (_ratio(r, base_bits), r.ndcg10),
                    textcoords="offset points", xytext=(6, 4), fontsize=9,
                    color="#e74c3c")
    ax.scatter([_ratio(r, base_bits) for r in angle],
               [r.ndcg10 for r in angle],
               color="#2ecc71", s=120, marker="^", zorder=5,
               label="Angle int uniform")
    for r in angle:
        b = r.bits_per_vec // (base_bits // 32 - 1)
        ax.annotate(f"int{b}", (_ratio(r, base_bits), r.ndcg10),
                    textcoords="offset points", xytext=(6, 4), fontsize=9,
                    color="#2ecc71")
    ax.axhline(base.ndcg10, color="black", linestyle="--", linewidth=1.2,
               label="float32 baseline")
    ax.set_xlabel("Compression ratio (vs float32)", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title("A vs B: Cartesian int vs Angle int at matched budgets\n"
                 "Same x = same storage cost", fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"plot -> {path}")


def plot_clipping(clip_pos: list[EvalResult], clip_jac: list[EvalResult],
                  greedy: list[EvalResult], base_ndcg: float,
                  base_bits: int, path: Path) -> None:
    def fracs(lst):
        return [int(m.group(1)) / 100.0
                for r in lst
                if (m := re.search(r"(\d+)%", r.label))]

    pf = fracs(clip_pos); jf = fracs(clip_jac)
    pn = [r.ndcg10 for r in clip_pos]
    jn = [r.ndcg10 for r in clip_jac]
    pb = [r.bits_per_vec for r in clip_pos]
    jb = [r.bits_per_vec for r in clip_jac]
    gr = [_ratio(r, base_bits) for r in greedy]
    gn = [r.ndcg10 for r in greedy]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: NDCG vs clipping fraction
    axes[0].plot(pf, pn, "o-", color="#e74c3c", lw=2, ms=8, label="Positional clip")
    axes[0].plot(jf, jn, "s-", color="#2ecc71", lw=2, ms=8, label="Jacobian clip")
    axes[0].axhline(base_ndcg, color="black", ls="--", lw=1.2, label="float32")
    axes[0].set_xlabel("Fraction of angles clipped (→ corpus mean)", fontsize=11)
    axes[0].set_ylabel("NDCG@10", fontsize=11)
    axes[0].set_title("C: Clipping fraction vs quality\n"
                      "green > red = Jacobian ranking helps", fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Middle: NDCG vs bits stored
    axes[1].plot(pb, pn, "o-", color="#e74c3c", lw=2, ms=8, label="Positional clip")
    axes[1].plot(jb, jn, "s-", color="#2ecc71", lw=2, ms=8, label="Jacobian clip")
    axes[1].axhline(base_ndcg, color="black", ls="--", lw=1.2, label="float32")
    axes[1].set_xlabel("Bits stored per vector", fontsize=11)
    axes[1].set_ylabel("NDCG@10", fontsize=11)
    axes[1].set_title("C: Bits stored vs quality\n"
                      "Same x = same storage cost", fontsize=10)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # Right: Jacobian clip vs Greedy tier (same budget axis)
    axes[2].plot([_ratio(r, base_bits) for r in clip_jac], jn,
                 "s-", color="#2ecc71", lw=2, ms=8, label="Jacobian clip (int8)")
    axes[2].plot(gr, gn, "D-", color="#3498db", lw=2, ms=8,
                 label="Greedy tier (mixed precision)")
    axes[2].axhline(base_ndcg, color="black", ls="--", lw=1.2, label="float32")
    axes[2].set_xlabel("Compression ratio (vs float32)", fontsize=11)
    axes[2].set_ylabel("NDCG@10", fontsize=11)
    axes[2].set_title("C vs D: Jacobian clip vs Greedy tier\n"
                      "Does mixed precision beat uniform int8?", fontsize=10)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Clipping & greedy tier: quality vs compression",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"plot -> {path}")


def plot_tradeoff(results: list[EvalResult], base_bits: int,
                  base_ndcg: float, path: Path) -> None:
    groups = {
        "cartesian":        ("Cartesian int",          "o",  "#e74c3c"),
        "angle-uniform":    ("Angle uniform",          "^",  "#27ae60"),
        "clip-positional":  ("Positional clip",        "v",  "#95a5a6"),
        "clip-jacobian":    ("Jacobian clip",          "s",  "#2ecc71"),
        "greedy-tier":      ("Greedy tier",            "D",  "#3498db"),
        "greedy+residual":  ("Greedy + residual int8", "X",  "#1abc9c"),
        "jac-trunc-int8":   ("Jac-trunc int8",        "P",  "#2980b9"),
        "jac-trunc-int4":   ("Jac-trunc int4",        "*",  "#9b59b6"),
        "jac-trunc-int2":   ("Jac-trunc int2",        "H",  "#f39c12"),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    for grp, (name, marker, color) in groups.items():
        pts = sorted(
            [(_ratio(r, base_bits), r.ndcg10)
             for r in results if r.group == grp and r.bits_per_vec > 0],
            key=lambda x: x[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        for ax in (ax1, ax2):
            ax.plot(xs, ys, marker=marker, label=name, color=color,
                    lw=1.5, ms=7)
    for ax in (ax1, ax2):
        ax.axhline(base_ndcg, color="black", ls="--", lw=1.2,
                   label="float32 baseline")
        ax.set_xlabel("Compression ratio (vs float32)", fontsize=12)
        ax.set_ylabel("NDCG@10", fontsize=12)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
    ax1.set_title("All schemes: compression vs NDCG (linear)", fontsize=11)
    ax2.set_title("All schemes: compression vs NDCG (log x)", fontsize=11)
    ax2.set_xscale("log")
    fig.suptitle("Embedding → Angles → Quantize: full tradeoff",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"plot -> {path}")


def plot_ablation(ablation: list[EvalResult], base_ndcg: float, path: Path) -> None:
    labels = [r.label.replace("[Ablation] ", "") for r in ablation]
    ndcgs  = [r.ndcg10 for r in ablation]
    colors = ["#e74c3c", "#f39c12", "#95a5a6", "#2ecc71",
              "#3498db", "#1abc9c"][:len(ablation)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(labels)), ndcgs, color=colors, edgecolor="white",
                   height=0.6)
    ax.axvline(base_ndcg, color="black", ls="--", lw=1.5, label="float32 baseline")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("NDCG@10", fontsize=12)
    ax.set_title(f"Ablation: marginal benefit of each component "
                 f"(~{ABLATION_BITS}b budget)", fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, ndcgs):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"plot -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir",   required=True)
    ap.add_argument("--data-dir",    required=True)
    ap.add_argument("--embed-tag",   default="")
    ap.add_argument("--max-corpus",  type=int, default=50_000,
                    help="Max corpus docs (default 50k = safe on 4 GB RAM).")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--output-dir",  default="eval_results")
    ap.add_argument("--batch-q",     type=int, default=256)
    ap.add_argument("--full-corpus-n", type=int, default=5_416_568,
                    help="Full corpus size for MB-saved calculation in tradeoff table.")
    args = ap.parse_args()

    embed_dir = Path(args.embed_dir).expanduser().resolve()
    data_dir  = Path(args.data_dir).expanduser().resolve()
    out_dir   = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {embed_dir / 'corpus'} ...")
    c_vecs, c_ids = load_split(embed_dir / "corpus", args.embed_tag,
                                args.max_corpus or None)
    print(f"  {c_vecs.shape[0]:,} docs  dim={c_vecs.shape[1]}")

    print(f"Loading queries from {embed_dir / 'queries'} ...")
    q_vecs, q_ids = load_split(embed_dir / "queries", args.embed_tag,
                                args.max_queries or None)
    print(f"  {q_vecs.shape[0]:,} queries")

    print(f"Loading qrels from {data_dir} ...")
    all_qrels = load_qrels(data_dir)

    c_id_set  = set(c_ids)
    filt_qids = [qid for qid in q_ids
                 if qid in all_qrels and
                 any(cid in c_id_set for cid in all_qrels[qid])]
    print(f"  {len(filt_qids):,} / {len(q_ids):,} queries have "
          f"relevant docs in corpus sample")
    if not filt_qids:
        print("ERROR: no queries with relevant docs. Increase --max-corpus.")
        return

    qid_idx  = {qid: i for i, qid in enumerate(q_ids)}
    q_vecs_f = q_vecs[[qid_idx[qid] for qid in filt_qids]]
    qrels_f  = {qid: all_qrels[qid] for qid in filt_qids}

    dim       = c_vecs.shape[1]
    base_bits = 32 * dim
    print(f"\nEval: {len(filt_qids):,} queries x {len(c_ids):,} corpus docs  "
          f"float32={base_bits:,}b/vec\n")

    results, clip_pos, clip_jac, greedy_sweep, ablation = run_all(
        c_vecs, q_vecs_f, c_ids, filt_qids, qrels_f, args.batch_q)

    base_ndcg = next(r.ndcg10 for r in results if "float32" in r.label)

    all_results = results + ablation   # everything for the CSV
    write_table(results,  out_dir / "results_table.txt",
                title="All schemes")
    write_ablation_table(ablation, base_ndcg, out_dir / "ablation_table.txt")
    write_tradeoff_table(clip_pos, clip_jac, base_ndcg,
                         args.full_corpus_n, dim,
                         out_dir / "tradeoff_table.txt")
    write_csv(all_results, out_dir / "results.csv")

    plot_main(results, out_dir / "plot_main.png")
    plot_clipping(clip_pos, clip_jac, greedy_sweep, base_ndcg, base_bits,
                  out_dir / "plot_clipping.png")
    plot_tradeoff(results, base_bits, base_ndcg, out_dir / "plot_tradeoff.png")
    plot_ablation(ablation, base_ndcg, out_dir / "plot_ablation.png")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
