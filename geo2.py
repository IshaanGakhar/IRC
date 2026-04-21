"""
Dynamic angle quantization with per-angle precision tiers.

Each angle in the hyperspherical representation is assigned one of six tiers
based on its measured sensitivity (how much quantization error there propagates
into reconstructed Cartesian coordinates):

    FLOAT32  -> 32 bits, exact
    FLOAT16  -> 16 bits, half precision
    UINT8    -> 8 bits,  uniform scalar on angle range
    UINT4    -> 4 bits
    UINT2    -> 2 bits
    CLIPPED  -> 0 bits,  angle reconstructed at its empirical mean

Assignment is greedy under a total-bit budget: upgrade the angle whose
expected-squared-reconstruction-error gain per added bit is largest.

Storage layout: each angle stored in its tier's native format, not packed into
uniform-width records. This means load time involves 6 slices/casts rather
than one contiguous read -- acceptable because the tier assignment is fixed
across the corpus.
"""

from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
N_ANGLES = EMBED_DIM - 1

TOP_K = 10

# Total bit budgets to sweep (per stored vector). float32 raw = 49,152 bits.
BIT_BUDGETS = [3072, 6144, 9216, 12288, 18432]   # 2x .. 16x compression


class Tier(IntEnum):
    CLIPPED = 0
    UINT2   = 1
    UINT4   = 2
    UINT8   = 3
    FLOAT16 = 4
    FLOAT32 = 5


TIER_BITS = {
    Tier.CLIPPED: 0,
    Tier.UINT2:   2,
    Tier.UINT4:   4,
    Tier.UINT8:   8,
    Tier.FLOAT16: 16,
    Tier.FLOAT32: 32,
}


# ---------------------------------------------------------------------------
# Corpus + queries (same as before)
# ---------------------------------------------------------------------------

CORPUS = [
    "The Python programming language was created by Guido van Rossum in 1991.",
    "Rust is a systems programming language focused on memory safety without a garbage collector.",
    "GPUs accelerate neural network training through massively parallel matrix multiplication.",
    "Transformers use self-attention to model long-range dependencies in sequences.",
    "Quantization reduces neural network precision to shrink model size and speed up inference.",
    "Retrieval-augmented generation combines dense vector search with language model generation.",
    "Vector databases index high-dimensional embeddings for approximate nearest neighbor search.",
    "HNSW is a graph-based index that provides logarithmic search time for ANN.",
    "Product quantization compresses vectors by partitioning and learning sub-codebooks.",
    "Cosine similarity measures the angle between vectors, ignoring their magnitude.",
    "Sourdough bread relies on wild yeast and lactobacillus for its characteristic tang.",
    "Italian carbonara is traditionally made with guanciale, pecorino, egg yolks, and black pepper.",
    "Miso is a fermented soybean paste central to Japanese cuisine and umami flavor.",
    "Kimchi is a Korean fermented vegetable dish, most commonly made from napa cabbage.",
    "Neapolitan pizza uses 00 flour and cooks in under ninety seconds in a wood-fired oven.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "Mitochondria are the energy-producing organelles found in most eukaryotic cells.",
    "The second law of thermodynamics states that entropy in an isolated system never decreases.",
    "Quantum entanglement links particles such that measuring one instantly affects the other.",
    "General relativity describes gravity as the curvature of spacetime caused by mass.",
    "The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 AD.",
    "The Industrial Revolution began in Britain in the late eighteenth century.",
    "The printing press was invented by Johannes Gutenberg around 1440.",
    "The fall of the Berlin Wall in 1989 marked the symbolic end of the Cold War.",
    "The Silk Road connected East Asia to the Mediterranean for over a thousand years.",
    "Jazz originated in New Orleans in the early twentieth century.",
    "Bach's Well-Tempered Clavier contains preludes and fugues in all twenty-four major and minor keys.",
    "The Beatles released Sgt. Pepper's Lonely Hearts Club Band in 1967.",
    "Hip hop emerged from block parties in the Bronx during the 1970s.",
    "A symphony orchestra is typically organized into strings, woodwinds, brass, and percussion sections.",
    "The marathon distance of 42.195 kilometers was standardized at the 1908 London Olympics.",
    "Cricket is played between two teams of eleven players on an oval field with a central pitch.",
    "Formula One cars can generate aerodynamic downforce exceeding their own weight at high speed.",
    "Basketball was invented by James Naismith in 1891 as an indoor winter sport.",
    "The FIFA World Cup is held every four years and is the most-watched sporting event globally.",
]

QUERIES = [
    ("Who invented Python?", 0),
    ("Programming language known for memory safety", 1),
    ("Why do GPUs speed up deep learning?", 2),
    ("What is self-attention in neural networks?", 3),
    ("How does lowering model precision help inference?", 4),
    ("What is RAG in language models?", 5),
    ("Database optimized for embedding search", 6),
    ("Graph-based approximate nearest neighbor algorithm", 7),
    ("Compressing vectors with codebooks", 8),
    ("Similarity metric that ignores vector length", 9),
    ("How is sourdough made?", 10),
    ("Authentic carbonara ingredients", 11),
    ("Japanese fermented paste for umami", 12),
    ("Korean fermented cabbage dish", 13),
    ("Traditional pizza from Naples", 14),
    ("How do plants make energy from light?", 15),
    ("Cellular powerhouse organelle", 16),
    ("Why does entropy always increase?", 17),
    ("Spooky action at a distance", 18),
    ("Einstein's theory of gravity", 19),
    ("When was the Roman Empire largest?", 20),
    ("Where did the Industrial Revolution start?", 21),
    ("Who invented the printing press?", 22),
    ("End of the Cold War symbol", 23),
    ("Ancient trade route between Asia and Europe", 24),
    ("Birthplace of jazz music", 25),
    ("Bach's famous keyboard work in all keys", 26),
    ("Beatles album from 1967", 27),
    ("Origin of hip hop culture", 28),
    ("Sections of an orchestra", 29),
    ("Why is a marathon 42 kilometers?", 30),
    ("Bat and ball sport with eleven players", 31),
    ("Downforce in motorsport", 32),
    ("Who invented basketball?", 33),
    ("World's biggest sporting tournament", 34),
]


# ---------------------------------------------------------------------------
# Embedding + hyperspherical conversion (unchanged)
# ---------------------------------------------------------------------------

def embed_texts(texts: Sequence[str]) -> np.ndarray:
    client = OpenAI()
    resp = client.embeddings.create(model=MODEL, input=list(texts))
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def cartesian_to_hyperspherical(x: np.ndarray) -> np.ndarray:
    N, n = x.shape
    angles = np.zeros((N, n - 1), dtype=np.float64)
    sq = x.astype(np.float64) ** 2
    rev_cumsum = np.cumsum(sq[:, ::-1], axis=1)[:, ::-1]
    tail = np.sqrt(np.clip(rev_cumsum, 0.0, None))
    for i in range(n - 2):
        denom = np.clip(tail[:, i], 1e-12, None)
        ratio = np.clip(x[:, i] / denom, -1.0, 1.0)
        angles[:, i] = np.arccos(ratio)
    last = np.arctan2(x[:, n - 1], x[:, n - 2])
    last = np.where(last < 0, last + 2 * math.pi, last)
    angles[:, n - 2] = last
    return angles


def hyperspherical_to_cartesian(angles: np.ndarray) -> np.ndarray:
    N, m = angles.shape
    n = m + 1
    x = np.zeros((N, n), dtype=np.float64)
    sin_prod = np.ones(N, dtype=np.float64)
    for i in range(m):
        x[:, i] = sin_prod * np.cos(angles[:, i])
        sin_prod = sin_prod * np.sin(angles[:, i])
    x[:, n - 1] = sin_prod
    return x


# ---------------------------------------------------------------------------
# Sensitivity estimation
# ---------------------------------------------------------------------------

def estimate_sensitivity(angles_calib: np.ndarray) -> np.ndarray:
    """
    Estimate per-angle sensitivity s_i := E[||dx/dtheta_i||^2] under the
    empirical distribution of the calibration set.

    For the standard hyperspherical parameterization:
        dx_j/dtheta_i = 0                              if j < i
                      = -prod_{k<i} sin(theta_k) * sin(theta_i)   if j = i
                      = prod_{k<i} sin(theta_k) * cos(theta_i) * (chain term) if j > i

    Taking the full ||dx/dtheta_i||^2 analytically is messy because of the
    chain structure. A clean, tight proxy that matches the dominant term is:

        s_i  proportional to  E[ prod_{k<i} sin(theta_k)^2 ]

    i.e., the expected squared sin-product up to angle i. This captures the
    fact that early angles propagate through every downstream coordinate at
    near full strength, while late angles have their influence damped by
    the accumulated sin(< 1) factors.

    We also multiply by E[range of theta_i quantization bin^2], which for
    a uniform bin of width w is w^2/12. But since bin width depends on the
    tier we're assigning, we factor it out and return only the prefactor.
    """
    N, M = angles_calib.shape
    # log-sin-squared, accumulated -- avoids underflow for long chains
    log_sin2 = 2.0 * np.log(np.clip(np.sin(angles_calib), 1e-12, None))   # (N, M)
    cum = np.zeros_like(log_sin2)
    cum[:, 1:] = np.cumsum(log_sin2[:, :-1], axis=1)
    # E over calibration set
    mean_log = cum.mean(axis=0)                                            # (M,)
    # Shift for numerical stability then exponentiate
    sens = np.exp(mean_log - mean_log.max())
    return sens   # relative, unnormalized


# ---------------------------------------------------------------------------
# Tier assignment (greedy knapsack)
# ---------------------------------------------------------------------------

def assign_tiers(sensitivity: np.ndarray, total_bits: int,
                 angle_ranges: np.ndarray) -> np.ndarray:
    """
    Assign each of M angles to a Tier under the bit budget `total_bits`.

    Objective: minimize sum_i sensitivity[i] * distortion(tier_i, range_i)
    where distortion is an approximation of expected squared reconstruction
    error at that precision:

        CLIPPED  : range^2 / 12     (angle replaced by midpoint; worst case var)
        UINT k   : (range / 2^k)^2 / 12   (uniform-scalar bin variance)
        FLOAT16  : ~epsilon * range   (relative eps_h ~ 9.8e-4)
        FLOAT32  : ~0

    Greedy algorithm:
        start everyone at CLIPPED, spend bits by repeatedly upgrading the
        angle with the best (distortion reduction) / (bit cost) ratio
        until the budget is exhausted.
    """
    M = len(sensitivity)

    def distortion(tier: Tier, r: float) -> float:
        if tier == Tier.CLIPPED:
            return (r * r) / 12.0
        if tier in (Tier.UINT2, Tier.UINT4, Tier.UINT8):
            bits = TIER_BITS[tier]
            w = r / (1 << bits)
            return (w * w) / 12.0
        if tier == Tier.FLOAT16:
            # worst-case half-float ULP over [0, r] ~ r * 2^-10
            return (r * (2.0 ** -10)) ** 2 / 12.0
        # FLOAT32: essentially zero distortion at our scales
        return (r * (2.0 ** -23)) ** 2 / 12.0

    # Start everyone clipped
    tiers = np.full(M, Tier.CLIPPED, dtype=np.int32)
    current_dist = np.array([sensitivity[i] * distortion(Tier.CLIPPED, angle_ranges[i])
                             for i in range(M)])
    used_bits = 0

    # Build the upgrade ladder once: sequence of tiers to step through
    ladder = [Tier.CLIPPED, Tier.UINT2, Tier.UINT4, Tier.UINT8, Tier.FLOAT16, Tier.FLOAT32]

    # For each angle, precompute the marginal gain (dist reduction) and cost (extra bits)
    # of each upgrade step. Store as list of (gain_per_bit, angle, new_tier) events;
    # sort descending.

    # Note: we can't do a simple priority queue because upgrading later creates
    # new upgrade options. Instead we iterate: at each step, find the best
    # single-step upgrade across all angles, apply it, repeat.

    # For speed on M=1535, maintain a "next upgrade cost/gain" per angle.
    def step_upgrade(i: int):
        cur = Tier(tiers[i])
        idx = ladder.index(cur)
        if idx == len(ladder) - 1:
            return None  # already top tier
        nxt = ladder[idx + 1]
        extra_bits = TIER_BITS[nxt] - TIER_BITS[cur]
        new_d = sensitivity[i] * distortion(nxt, angle_ranges[i])
        gain = current_dist[i] - new_d
        return (gain / extra_bits, extra_bits, nxt, new_d)

    # Initial upgrade options
    opts = [step_upgrade(i) for i in range(M)]

    # Greedy loop
    while used_bits < total_bits:
        # best option
        best_i = -1
        best_ratio = -np.inf
        best_tuple = None
        for i, o in enumerate(opts):
            if o is None:
                continue
            if o[1] > total_bits - used_bits:   # wouldn't fit
                continue
            if o[0] > best_ratio:
                best_ratio = o[0]
                best_i = i
                best_tuple = o
        if best_i < 0 or best_tuple is None or best_ratio <= 0:
            break  # no beneficial upgrade fits
        _, extra_bits, new_tier, new_d = best_tuple
        tiers[best_i] = int(new_tier)
        current_dist[best_i] = new_d
        used_bits += extra_bits
        opts[best_i] = step_upgrade(best_i)

    return tiers


# ---------------------------------------------------------------------------
# Quantize / dequantize per tier
# ---------------------------------------------------------------------------

@dataclass
class QuantizedAngles:
    """Holds per-tier arrays + the tier assignment + calibration stats."""
    tiers: np.ndarray           # (M,) int
    f32_idx: np.ndarray         # indices of angles in FLOAT32 tier
    f32_vals: np.ndarray        # (N, len(f32_idx))
    f16_idx: np.ndarray
    f16_vals: np.ndarray        # stored as float16
    u8_idx: np.ndarray
    u8_vals: np.ndarray         # uint8
    u4_idx: np.ndarray
    u4_vals: np.ndarray         # uint8 holding values 0..15
    u2_idx: np.ndarray
    u2_vals: np.ndarray         # uint8 holding values 0..3
    clipped_means: np.ndarray   # (M,) replacement for clipped angles
    angle_ranges: np.ndarray    # (M,) range of each angle [0,pi] or [0,2pi]

    def bits_per_vec(self) -> int:
        t = self.tiers
        return int(
            32 * np.sum(t == Tier.FLOAT32) +
            16 * np.sum(t == Tier.FLOAT16) +
             8 * np.sum(t == Tier.UINT8)   +
             4 * np.sum(t == Tier.UINT4)   +
             2 * np.sum(t == Tier.UINT2)
        )


def quantize_dynamic(angles: np.ndarray, tiers: np.ndarray,
                     clipped_means: np.ndarray,
                     angle_ranges: np.ndarray) -> QuantizedAngles:
    def idx_of(tier):
        return np.where(tiers == int(tier))[0]

    f32_idx = idx_of(Tier.FLOAT32)
    f16_idx = idx_of(Tier.FLOAT16)
    u8_idx  = idx_of(Tier.UINT8)
    u4_idx  = idx_of(Tier.UINT4)
    u2_idx  = idx_of(Tier.UINT2)

    f32_vals = angles[:, f32_idx].astype(np.float32)
    f16_vals = angles[:, f16_idx].astype(np.float16)

    def uquant(a, bits, r):
        levels = 1 << bits
        a_clip = np.clip(a, 0.0, r - 1e-12)
        q = np.floor(a_clip / r * levels).astype(np.int32)
        return np.clip(q, 0, levels - 1).astype(np.uint8)

    u8_vals = uquant(angles[:, u8_idx], 8, angle_ranges[u8_idx][None, :]) if len(u8_idx) else np.zeros((angles.shape[0], 0), dtype=np.uint8)
    u4_vals = uquant(angles[:, u4_idx], 4, angle_ranges[u4_idx][None, :]) if len(u4_idx) else np.zeros((angles.shape[0], 0), dtype=np.uint8)
    u2_vals = uquant(angles[:, u2_idx], 2, angle_ranges[u2_idx][None, :]) if len(u2_idx) else np.zeros((angles.shape[0], 0), dtype=np.uint8)

    return QuantizedAngles(
        tiers=tiers,
        f32_idx=f32_idx, f32_vals=f32_vals,
        f16_idx=f16_idx, f16_vals=f16_vals,
        u8_idx=u8_idx,   u8_vals=u8_vals,
        u4_idx=u4_idx,   u4_vals=u4_vals,
        u2_idx=u2_idx,   u2_vals=u2_vals,
        clipped_means=clipped_means,
        angle_ranges=angle_ranges,
    )


def dequantize_dynamic(q: QuantizedAngles, N: int) -> np.ndarray:
    M = len(q.tiers)
    out = np.empty((N, M), dtype=np.float64)

    # Start with clipped means for ALL angles; overwrite others from stored tiers.
    # Broadcasting the (M,) mean vector over N rows.
    out[:] = q.clipped_means[None, :]

    if len(q.f32_idx):
        out[:, q.f32_idx] = q.f32_vals.astype(np.float64)
    if len(q.f16_idx):
        out[:, q.f16_idx] = q.f16_vals.astype(np.float64)

    def udequant(codes, bits, r_vec):
        levels = 1 << bits
        return (codes.astype(np.float64) + 0.5) * (r_vec[None, :] / levels)

    if len(q.u8_idx):
        out[:, q.u8_idx] = udequant(q.u8_vals, 8, q.angle_ranges[q.u8_idx])
    if len(q.u4_idx):
        out[:, q.u4_idx] = udequant(q.u4_vals, 4, q.angle_ranges[q.u4_idx])
    if len(q.u2_idx):
        out[:, q.u2_idx] = udequant(q.u2_vals, 2, q.angle_ranges[q.u2_idx])

    return out


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def cosine_topk(queries: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    row_scores = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-row_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1)


@dataclass
class RetrievalStats:
    label: str
    bits_per_vec: int
    recall_at_1: float
    recall_at_k: float
    mean_rank_of_truth: float
    score_time_ms: float
    tier_breakdown: str = ""


def evaluate(label: str, bits_per_vec: int,
             query_unit: np.ndarray, corpus_unit: np.ndarray,
             truth: list[int], k: int, tier_breakdown: str = "") -> RetrievalStats:
    t0 = time.perf_counter()
    top = cosine_topk(query_unit, corpus_unit, k)
    sims_all = query_unit @ corpus_unit.T
    t1 = time.perf_counter()

    truth_arr = np.array(truth)
    hit1 = np.mean(top[:, 0] == truth_arr)
    hitk = np.mean(np.any(top == truth_arr[:, None], axis=1))
    order = np.argsort(-sims_all, axis=1)
    ranks = np.array([np.where(order[i] == truth[i])[0][0] + 1 for i in range(len(truth))])

    return RetrievalStats(
        label=label, bits_per_vec=bits_per_vec,
        recall_at_1=float(hit1), recall_at_k=float(hitk),
        mean_rank_of_truth=float(np.mean(ranks)),
        score_time_ms=(t1 - t0) * 1000.0,
        tier_breakdown=tier_breakdown,
    )


def tier_summary(tiers: np.ndarray) -> str:
    c = {t: int(np.sum(tiers == int(t))) for t in Tier}
    return (f"f32={c[Tier.FLOAT32]} f16={c[Tier.FLOAT16]} "
            f"u8={c[Tier.UINT8]} u4={c[Tier.UINT4]} "
            f"u2={c[Tier.UINT2]} clip={c[Tier.CLIPPED]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    print(f"Embedding {len(CORPUS)} docs + {len(QUERIES)} queries via {MODEL} ...")
    doc_vecs = embed_texts(CORPUS)
    q_texts = [q for q, _ in QUERIES]
    truth = [t for _, t in QUERIES]
    q_vecs = embed_texts(q_texts)
    print(f"  doc shape={doc_vecs.shape}  query shape={q_vecs.shape}")

    # Baselines
    baseline = evaluate("float32 raw", 32 * EMBED_DIM, q_vecs, doc_vecs, truth, TOP_K)

    scale = 127.0
    doc_i8 = np.round(np.clip(doc_vecs, -1, 1) * scale).astype(np.int8)
    q_i8   = np.round(np.clip(q_vecs,   -1, 1) * scale).astype(np.int8)
    doc_i8_f = doc_i8.astype(np.float32) / scale
    q_i8_f   = q_i8.astype(np.float32)   / scale
    doc_i8_f /= np.clip(np.linalg.norm(doc_i8_f, axis=1, keepdims=True), 1e-12, None)
    q_i8_f   /= np.clip(np.linalg.norm(q_i8_f,   axis=1, keepdims=True), 1e-12, None)
    int8_stats = evaluate("int8 per-dim", 8 * EMBED_DIM, q_i8_f, doc_i8_f, truth, TOP_K)

    # Convert to angles
    print("\nConverting to hyperspherical angles ...")
    doc_angles = cartesian_to_hyperspherical(doc_vecs)
    q_angles   = cartesian_to_hyperspherical(q_vecs)

    rt = hyperspherical_to_cartesian(doc_angles)
    rt_err = float(np.mean(np.linalg.norm(rt - doc_vecs, axis=1)))
    print(f"  round-trip mean L2 error: {rt_err:.3e}")

    # Calibration: sensitivity + mean per angle + range
    sens = estimate_sensitivity(doc_angles)
    angle_means = doc_angles.mean(axis=0)
    angle_ranges = np.full(N_ANGLES, math.pi, dtype=np.float64)
    angle_ranges[-1] = 2 * math.pi

    print(f"\nSensitivity profile (higher = more impact on reconstruction):")
    print(f"  top-5 angles: indices={np.argsort(-sens)[:5].tolist()}  "
          f"values={sorted(sens, reverse=True)[:5]}")
    print(f"  mean of early 100 sens: {sens[:100].mean():.3e}")
    print(f"  mean of late 100 sens:  {sens[-100:].mean():.3e}")

    results: list[RetrievalStats] = [baseline, int8_stats]

    print("\n=== Dynamic tier quantization sweep ===")
    for budget in BIT_BUDGETS:
        tiers = assign_tiers(sens, budget, angle_ranges)

        q_doc = quantize_dynamic(doc_angles, tiers, angle_means, angle_ranges)
        q_q   = quantize_dynamic(q_angles,   tiers, angle_means, angle_ranges)

        doc_deq = dequantize_dynamic(q_doc, doc_angles.shape[0])
        q_deq   = dequantize_dynamic(q_q,   q_angles.shape[0])

        doc_recon = hyperspherical_to_cartesian(doc_deq).astype(np.float32)
        q_recon   = hyperspherical_to_cartesian(q_deq).astype(np.float32)
        doc_recon /= np.clip(np.linalg.norm(doc_recon, axis=1, keepdims=True), 1e-12, None)
        q_recon   /= np.clip(np.linalg.norm(q_recon,   axis=1, keepdims=True), 1e-12, None)

        actual_bits = q_doc.bits_per_vec()
        stats = evaluate(
            label=f"dyn-tier budget={budget:>5}b",
            bits_per_vec=actual_bits,
            query_unit=q_recon, corpus_unit=doc_recon,
            truth=truth, k=TOP_K,
            tier_breakdown=tier_summary(tiers),
        )
        results.append(stats)

        # Also asymmetric variant: float32 query, tier-quantized docs
        asym = evaluate(
            label=f"dyn-tier budget={budget:>5}b ASYM(q=f32)",
            bits_per_vec=actual_bits,
            query_unit=q_vecs, corpus_unit=doc_recon,
            truth=truth, k=TOP_K,
            tier_breakdown=tier_summary(tiers),
        )
        results.append(asym)

    # Report
    print("\n" + "=" * 120)
    print(f"{'method':<32} {'bits':>8} {'R@1':>6} {'R@'+str(TOP_K):>6} "
          f"{'meanRk':>8} {'ms':>7}   tier breakdown")
    print("-" * 120)
    for r in results:
        print(f"{r.label:<32} {r.bits_per_vec:>8d} "
              f"{r.recall_at_1:>6.3f} {r.recall_at_k:>6.3f} "
              f"{r.mean_rank_of_truth:>8.2f} {r.score_time_ms:>7.3f}   {r.tier_breakdown}")
    print("=" * 120)

    print("\nCompression vs float32 baseline:")
    base_bits = results[0].bits_per_vec
    for r in results[1:]:
        print(f"  {base_bits / r.bits_per_vec:>5.1f}x   {r.label}")


if __name__ == "__main__":
    main()