"""
Angle-quantized retrieval vs. baseline float32 cosine similarity.

Pipeline:
  1. Embed a corpus + query set with text-embedding-3-small (1536-dim, L2-normalized).
  2. Convert each unit vector on S^{1535} to hyperspherical angles (1535 angles).
  3. Quantize angles with Jacobian-aware (non-uniform) bit allocation at several budgets.
  4. Compare retrieval recall@k and mean-average-rank-error vs. the float32 ground truth.
  5. Report storage per vector and scoring throughput.

Key design notes:
  - text-embedding-3-small returns unit-norm vectors already.
  - In hyperspherical coords (x_1..x_n), angle theta_i has Jacobian weight
    sin(theta_1)^{n-2} * sin(theta_2)^{n-3} * ... so earlier angles dominate
    the spherical measure. Bit allocation is proportional to log(expected
    chord-length contribution) of each angle under a uniform-on-sphere prior.
  - Distance reconstruction: we dequantize angles, rebuild the unit vector,
    compute cosine. This is NOT the fast-scan LUT path from the earlier
    discussion -- it's the reference implementation to measure quality loss.
    Speed numbers therefore reflect quality-preserving decode, not the
    ultimate SIMD-LUT path.

Usage:
  export OPENAI_API_KEY=sk-...
  python3 angle_quant_benchmark.py
"""

from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass
from typing import Sequence
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "text-embedding-3-small"   # 1536 dims, L2-normalized
EMBED_DIM = 1536
N_ANGLES = EMBED_DIM - 1           # hyperspherical parameterization

# Bit budgets to sweep (total bits per stored vector, excluding sign of unit-norm).
# float32 baseline = 32 * 1536 = 49,152 bits. int8-per-dim = 12,288 bits.
# Interesting regime is 1500-12000 bits where quality-compression tradeoff lives.
BIT_BUDGETS = [1536, 3072, 4608, 6144, 9216, 12288]   # 1-8 avg bits/angle

TOP_K = 10                          # recall@k
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# A small, topically diverse corpus. Queries are paraphrases / near-matches
# so we can check whether the expected document rises to the top.
CORPUS = [
    # Tech
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
    # Food
    "Sourdough bread relies on wild yeast and lactobacillus for its characteristic tang.",
    "Italian carbonara is traditionally made with guanciale, pecorino, egg yolks, and black pepper.",
    "Miso is a fermented soybean paste central to Japanese cuisine and umami flavor.",
    "Kimchi is a Korean fermented vegetable dish, most commonly made from napa cabbage.",
    "Neapolitan pizza uses 00 flour and cooks in under ninety seconds in a wood-fired oven.",
    # Science
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "Mitochondria are the energy-producing organelles found in most eukaryotic cells.",
    "The second law of thermodynamics states that entropy in an isolated system never decreases.",
    "Quantum entanglement links particles such that measuring one instantly affects the other.",
    "General relativity describes gravity as the curvature of spacetime caused by mass.",
    # History
    "The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 AD.",
    "The Industrial Revolution began in Britain in the late eighteenth century.",
    "The printing press was invented by Johannes Gutenberg around 1440.",
    "The fall of the Berlin Wall in 1989 marked the symbolic end of the Cold War.",
    "The Silk Road connected East Asia to the Mediterranean for over a thousand years.",
    # Music
    "Jazz originated in New Orleans in the early twentieth century.",
    "Bach's Well-Tempered Clavier contains preludes and fugues in all twenty-four major and minor keys.",
    "The Beatles released Sgt. Pepper's Lonely Hearts Club Band in 1967.",
    "Hip hop emerged from block parties in the Bronx during the 1970s.",
    "A symphony orchestra is typically organized into strings, woodwinds, brass, and percussion sections.",
    # Sports
    "The marathon distance of 42.195 kilometers was standardized at the 1908 London Olympics.",
    "Cricket is played between two teams of eleven players on an oval field with a central pitch.",
    "Formula One cars can generate aerodynamic downforce exceeding their own weight at high speed.",
    "Basketball was invented by James Naismith in 1891 as an indoor winter sport.",
    "The FIFA World Cup is held every four years and is the most-watched sporting event globally.",
]

# Each query is semantically aligned with one doc index (ground truth).
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
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: Sequence[str]) -> np.ndarray:
    client = OpenAI()
    # Batch in one call; API handles up to 2048 inputs per request.
    resp = client.embeddings.create(model=MODEL, input=list(texts))
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # text-embedding-3-small returns unit vectors, but re-normalize defensively.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-12, None)
    return vecs


# ---------------------------------------------------------------------------
# Hyperspherical conversion
# ---------------------------------------------------------------------------

def cartesian_to_hyperspherical(x: np.ndarray) -> np.ndarray:
    """
    Convert unit vectors x of shape (N, n) to hyperspherical angles of shape (N, n-1).

    Convention (standard):
        x_1     = cos(theta_1)
        x_2     = sin(theta_1) cos(theta_2)
        ...
        x_{n-1} = sin(theta_1) ... sin(theta_{n-2}) cos(theta_{n-1})
        x_n     = sin(theta_1) ... sin(theta_{n-2}) sin(theta_{n-1})

    theta_1 .. theta_{n-2} in [0, pi]
    theta_{n-1}            in [0, 2*pi)   (uses atan2 on the last two coords)
    """
    N, n = x.shape
    angles = np.zeros((N, n - 1), dtype=np.float64)

    # tail[i] = sqrt(x_{i+1}^2 + ... + x_n^2), for i = 0 .. n-1
    # tail[0] = 1 for unit vectors.
    sq = x.astype(np.float64) ** 2
    # cumulative sum from the right
    rev_cumsum = np.cumsum(sq[:, ::-1], axis=1)[:, ::-1]
    tail = np.sqrt(np.clip(rev_cumsum, 0.0, None))   # shape (N, n)

    # theta_i = arccos(x_i / tail_i) for i = 1..n-2  (0-indexed: 0..n-3)
    for i in range(n - 2):
        denom = np.clip(tail[:, i], 1e-12, None)
        ratio = np.clip(x[:, i] / denom, -1.0, 1.0)
        angles[:, i] = np.arccos(ratio)

    # Last angle uses atan2 so it covers [0, 2*pi).
    last = np.arctan2(x[:, n - 1], x[:, n - 2])
    # Shift into [0, 2*pi)
    last = np.where(last < 0, last + 2 * math.pi, last)
    angles[:, n - 2] = last

    return angles


def hyperspherical_to_cartesian(angles: np.ndarray) -> np.ndarray:
    """Inverse of the above. angles shape (N, n-1) -> x shape (N, n)."""
    N, m = angles.shape
    n = m + 1
    x = np.zeros((N, n), dtype=np.float64)

    sin_prod = np.ones(N, dtype=np.float64)
    for i in range(m):
        x[:, i] = sin_prod * np.cos(angles[:, i])
        sin_prod = sin_prod * np.sin(angles[:, i])
    x[:, n - 1] = sin_prod   # final coord is just the accumulated sin product
    # (This is sin(theta_1)*...*sin(theta_{n-2})*sin(theta_{n-1}),
    #  which matches the convention above because we used cos for x_{n-1}
    #  via the loop's last iteration and sin stays in the product.)
    return x


# ---------------------------------------------------------------------------
# Jacobian-aware bit allocation
# ---------------------------------------------------------------------------

def jacobian_bit_allocation(n_angles: int, total_bits: int,
                            min_bits: int = 0, max_bits: int = 16,
                            empirical_std: np.ndarray | None = None) -> np.ndarray:
    """
    Allocate bits per angle. Two modes:

      1. Propagation-weighted prior (default): an error in theta_i perturbs
         Cartesian coords x_i, x_{i+1}, ..., x_n via the sin-product chain.
         Early angles affect MORE coordinates, so they need MORE bits.
         Weight: sqrt(n_angles - i)  (number of downstream coords).

      2. Empirical (preferred when a calibration sample is available):
         pass in per-angle std estimated from real embeddings, and allocate
         bits proportional to log2(std). This is the rate-distortion-correct
         allocation assuming Gaussian-ish marginals.

    min_bits=0 allows dropping angles entirely; the dequantizer reconstructs
    0-bit angles at the midpoint of their range (min-distortion under uniform
    prior). This matters when total_bits < n_angles -- the truncation regime.
    """
    if empirical_std is not None:
        # log2(std) can be negative for tightly-concentrated angles; shift
        # so the minimum is ~0, then add an epsilon so every angle has some
        # weight before the fix-up step redistributes.
        log_std = np.log2(np.clip(empirical_std, 1e-6, None))
        importance = log_std - log_std.min() + 0.1
    else:
        importance = np.zeros(n_angles, dtype=np.float64)
        # theta_i propagates into coords i, i+1, ..., n-1 -- that's n_angles - i
        # Cartesian components (counting the last one via the sin-product tail).
        for i in range(n_angles):
            importance[i] = math.sqrt(n_angles - i)

    weights = importance / importance.sum()

    # Continuous ideal allocation:
    ideal = weights * total_bits

    # Start at floor, clipped to [min_bits, max_bits]
    bits = np.clip(np.floor(ideal).astype(np.int32), min_bits, max_bits)

    # Fix-up: add or remove bits one at a time, greedy.
    remaining = int(total_bits - bits.sum())
    # residual "hunger" = ideal - current
    # when adding bits, give to the most hungry; when removing, take from least hungry
    guard = 0
    max_iters = n_angles * max_bits * 4   # safety
    while remaining != 0 and guard < max_iters:
        residual = ideal - bits
        if remaining > 0:
            # Pick the angle with largest positive residual that still has headroom
            mask = bits < max_bits
            if not mask.any():
                break
            res_masked = np.where(mask, residual, -np.inf)
            j = int(np.argmax(res_masked))
            bits[j] += 1
            remaining -= 1
        else:
            mask = bits > min_bits
            if not mask.any():
                break
            # Take from the angle with the most negative residual (over-allocated)
            res_masked = np.where(mask, residual, np.inf)
            j = int(np.argmin(res_masked))
            bits[j] -= 1
            remaining += 1
        guard += 1
    return bits


# ---------------------------------------------------------------------------
# Quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_angles(angles: np.ndarray, bits_per_angle: np.ndarray) -> np.ndarray:
    """
    Uniform scalar quantization per angle dimension (vectorized).
      - Angles 0..n-2 live in [0, pi]
      - Angle n-1 lives in [0, 2*pi)
    Returns uint32 codes of shape (N, n_angles). Bit widths vary per column.
    """
    M = angles.shape[1]
    hi = np.full(M, math.pi, dtype=np.float64)
    hi[-1] = 2 * math.pi
    levels = (1 << bits_per_angle.astype(np.int64))            # (M,)
    a = np.clip(angles, 0.0, hi - 1e-12)                        # (N, M)
    q = np.floor(a / hi * levels).astype(np.int64)
    np.clip(q, 0, levels - 1, out=q)
    return q.astype(np.uint32)


def dequantize_angles(codes: np.ndarray, bits_per_angle: np.ndarray) -> np.ndarray:
    M = codes.shape[1]
    hi = np.full(M, math.pi, dtype=np.float64)
    hi[-1] = 2 * math.pi
    levels = (1 << bits_per_angle.astype(np.int64))
    return (codes.astype(np.float64) + 0.5) * (hi / levels)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def cosine_topk(queries: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices per query (both inputs are unit vectors)."""
    sims = queries @ corpus.T
    # argsort descending
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    # sort those k
    row_scores = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-row_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1)


@dataclass
class RetrievalStats:
    label: str
    bits_per_vec: int
    recall_at_1: float
    recall_at_k: float
    mean_rank_of_truth: float      # lower is better; 1.0 is perfect
    score_time_ms: float


def evaluate(label: str, bits_per_vec: int,
             query_unit: np.ndarray, corpus_unit: np.ndarray,
             truth: list[int], k: int) -> RetrievalStats:
    t0 = time.perf_counter()
    top = cosine_topk(query_unit, corpus_unit, k)
    sims_all = query_unit @ corpus_unit.T   # for rank-of-truth
    t1 = time.perf_counter()

    truth_arr = np.array(truth)
    hit1 = np.mean(top[:, 0] == truth_arr)
    hitk = np.mean(np.any(top == truth_arr[:, None], axis=1))

    # Rank of the true doc per query (1 = top)
    order = np.argsort(-sims_all, axis=1)
    ranks = np.array([np.where(order[i] == truth[i])[0][0] + 1
                      for i in range(len(truth))])
    mean_rank = float(np.mean(ranks))

    return RetrievalStats(
        label=label,
        bits_per_vec=bits_per_vec,
        recall_at_1=float(hit1),
        recall_at_k=float(hitk),
        mean_rank_of_truth=mean_rank,
        score_time_ms=(t1 - t0) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _quantize_pipeline(angles_docs, angles_q, bits):
    """Run quantize -> dequantize -> reconstruct -> renormalize. Returns (doc_recon, q_recon)."""
    doc_codes = quantize_angles(angles_docs, bits)
    q_codes   = quantize_angles(angles_q, bits)
    doc_deq = dequantize_angles(doc_codes, bits)
    q_deq   = dequantize_angles(q_codes, bits)
    doc_recon = hyperspherical_to_cartesian(doc_deq).astype(np.float32)
    q_recon   = hyperspherical_to_cartesian(q_deq).astype(np.float32)
    doc_recon /= np.clip(np.linalg.norm(doc_recon, axis=1, keepdims=True), 1e-12, None)
    q_recon   /= np.clip(np.linalg.norm(q_recon,   axis=1, keepdims=True), 1e-12, None)
    return doc_recon, q_recon


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running.")

    print(f"Embedding {len(CORPUS)} docs + {len(QUERIES)} queries via {MODEL} ...")
    doc_vecs = embed_texts(CORPUS)
    query_texts = [q for q, _ in QUERIES]
    truth = [t for _, t in QUERIES]
    q_vecs = embed_texts(query_texts)
    print(f"  doc shape={doc_vecs.shape}  query shape={q_vecs.shape}")

    # ---- Baseline: float32 ----
    print("\n=== Baseline: float32 cosine ===")
    baseline = evaluate(
        label="float32 raw",
        bits_per_vec=32 * EMBED_DIM,
        query_unit=q_vecs,
        corpus_unit=doc_vecs,
        truth=truth,
        k=TOP_K,
    )

    # ---- Baseline: int8-per-dim scalar quantization (standard compression baseline) ----
    scale = 127.0
    doc_i8 = np.round(np.clip(doc_vecs, -1, 1) * scale).astype(np.int8)
    q_i8   = np.round(np.clip(q_vecs,   -1, 1) * scale).astype(np.int8)
    doc_i8_f = (doc_i8.astype(np.float32) / scale)
    q_i8_f   = (q_i8.astype(np.float32)   / scale)
    doc_i8_f /= np.clip(np.linalg.norm(doc_i8_f, axis=1, keepdims=True), 1e-12, None)
    q_i8_f   /= np.clip(np.linalg.norm(q_i8_f,   axis=1, keepdims=True), 1e-12, None)
    int8_stats = evaluate(
        label="int8 per-dim",
        bits_per_vec=8 * EMBED_DIM,
        query_unit=q_i8_f,
        corpus_unit=doc_i8_f,
        truth=truth,
        k=TOP_K,
    )

    # ---- Convert to angles ----
    print("\nConverting to hyperspherical angles ...")
    doc_angles = cartesian_to_hyperspherical(doc_vecs)
    q_angles   = cartesian_to_hyperspherical(q_vecs)

    reconstructed = hyperspherical_to_cartesian(doc_angles)
    rt_err = float(np.mean(np.linalg.norm(reconstructed - doc_vecs, axis=1)))
    print(f"  round-trip mean L2 error: {rt_err:.3e}  (should be <1e-5)")

    # Empirical per-angle std over the corpus. With only ~35 docs this is
    # noisy; in real use you'd calibrate on 10k+ vectors.
    emp_std = doc_angles.std(axis=0)
    print(f"  empirical angle std: min={emp_std.min():.3f}  "
          f"median={float(np.median(emp_std)):.3f}  max={emp_std.max():.3f}")

    # ---- Sweep budgets x schemes ----
    results: list[RetrievalStats] = [baseline, int8_stats]

    schemes = [
        ("prop",  None,    "propagation-weighted"),
        ("emp",   emp_std, "empirical-std"),
    ]

    print("\n=== Angle-quantized sweep ===")
    for budget in BIT_BUDGETS:
        for tag, emp, long_name in schemes:
            bits = jacobian_bit_allocation(N_ANGLES, budget, empirical_std=emp)
            doc_recon, q_recon = _quantize_pipeline(doc_angles, q_angles, bits)

            stats = evaluate(
                label=f"angle-quant {budget:>5}b [{long_name}]  "
                      f"min/med/max={int(bits.min())}/{int(np.median(bits))}/{int(bits.max())}",
                bits_per_vec=int(bits.sum()),
                query_unit=q_recon,
                corpus_unit=doc_recon,
                truth=truth,
                k=TOP_K,
            )
            results.append(stats)

    # ---- Asymmetric variant: query stays float32, docs quantized (propagation scheme) ----
    mid_budget = BIT_BUDGETS[len(BIT_BUDGETS) // 2]
    bits = jacobian_bit_allocation(N_ANGLES, mid_budget)
    doc_codes = quantize_angles(doc_angles, bits)
    doc_recon = hyperspherical_to_cartesian(
        dequantize_angles(doc_codes, bits)
    ).astype(np.float32)
    doc_recon /= np.clip(np.linalg.norm(doc_recon, axis=1, keepdims=True), 1e-12, None)

    asym = evaluate(
        label=f"angle-quant {mid_budget}b [propagation, ASYMMETRIC query=float32]",
        bits_per_vec=int(bits.sum()),
        query_unit=q_vecs,
        corpus_unit=doc_recon,
        truth=truth,
        k=TOP_K,
    )
    results.append(asym)

    # ---- Report ----
    print("\n" + "=" * 110)
    print(f"{'method':<70} {'bits/vec':>10} {'R@1':>6} {'R@'+str(TOP_K):>6} "
          f"{'meanRk':>8} {'ms':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r.label:<70} {r.bits_per_vec:>10d} "
              f"{r.recall_at_1:>6.3f} {r.recall_at_k:>6.3f} "
              f"{r.mean_rank_of_truth:>8.2f} {r.score_time_ms:>8.3f}")
    print("=" * 110)

    print("\nCompression ratio vs. float32:")
    base_bits = results[0].bits_per_vec
    for r in results[1:]:
        print(f"  {base_bits / r.bits_per_vec:>5.1f}x   {r.label}")

    print(f"\nNote: float32 corpus = {len(CORPUS) * EMBED_DIM * 4 / 1024:.1f} KB total; "
          f"angle-quant at 6144b = {len(CORPUS) * 6144 / 8 / 1024:.1f} KB total.")


if __name__ == "__main__":
    main()