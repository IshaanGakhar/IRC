"""
Quantization evaluation on real BEIR embeddings.

Pipeline under test:
    embedding (float32) -> hyperspherical angles -> quantize angles (int8 / int4 / int2 / 0-bit)

Four questions:
  1. Accuracy:   angle->int8 vs embedding->int8 (same budget, which representation wins?)
  2. Redundancy: is int4/int2 a meaningful step over int8, or are intermediate precisions
                 made redundant by truncation (0-bit clipping of low-sensitivity angles)?
  3. Clipping:   0-bit angles are replaced by the corpus mean at query time -- sweep the
                 fraction of angles clipped to find the accuracy cliff.
  4. Tradeoff:   compression ratio vs NDCG@10 plot across all schemes.

Inputs:
  --embed-dir   chunked embedding root for one dataset (e.g. embeddings/fever)
                expects corpus/ and queries/ sub-dirs with chunk_*.npy files.
  --data-dir    BEIR dataset dir (e.g. bier-data/fever), used to load qrels.
  --embed-tag   chunk filename tag ("" for OAI plain layout, "minilm-l6-vllm" etc.)

Memory:
  Full FEVER (5.4M x 1536 x fp32) ~33 GB. Use --max-corpus to subsample.
  Default 50k is safe on t3.medium (4 GB). Use a large-RAM machine for full-corpus.

Outputs (in --output-dir):
  results_table.txt    human-readable metrics table
  results.csv          machine-readable
  plot_main.png        Cartesian vs Angle at int8/int4/int2 (TODO 1 & 2)
  plot_clipping.png    NDCG@10 vs fraction of angles clipped (TODO 3)
  plot_tradeoff.png    compression ratio vs NDCG@10 for all schemes (TODO 4)
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
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
            f"No chunks found in {split_dir} with tag='{tag}'. "
            f"Check --embed-tag and that embeddings exist.")
    all_vecs: list[np.ndarray] = []
    all_ids:  list[str] = []
    for p in chunks:
        arr  = np.load(p)
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
        vecs, all_ids = vecs[:max_vecs], all_ids[:max_vecs]
    return vecs, all_ids


# ---------------------------------------------------------------------------
# Qrels loading
# ---------------------------------------------------------------------------

def load_qrels(data_dir: Path) -> dict[str, dict[str, int]]:
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
            qid, cid, score = row["query-id"], row["corpus-id"], int(row["score"])
            if score > 0:
                qrels.setdefault(qid, {})[cid] = score
    return qrels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dcg(relevances: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(top_ids: np.ndarray, relevant: dict[str, int], k: int) -> float:
    rels  = [relevant.get(cid, 0) for cid in top_ids[:k]]
    ideal = sorted(relevant.values(), reverse=True)
    idcg  = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def recall_at_k(top_ids: np.ndarray, relevant: dict[str, int], k: int) -> float:
    hits = sum(1 for cid in top_ids[:k] if cid in relevant)
    return hits / len(relevant) if relevant else 0.0


@dataclass
class EvalResult:
    label: str
    group: str          # cartesian | angle-uniform | angle-truncation | clipping
    bits_per_vec: int
    ndcg10: float
    recall100: float
    score_ms: float


def evaluate(
    label: str, group: str, bits_per_vec: int,
    q_vecs: np.ndarray, c_vecs: np.ndarray,
    q_ids: list[str], c_ids: list[str],
    qrels: dict[str, dict[str, int]],
    batch_q: int = 256,
) -> EvalResult:
    c_ids_arr = np.array(c_ids)
    top_k = 100
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    t0 = time.perf_counter()
    for qi in range(0, len(q_ids), batch_q):
        q_batch = q_vecs[qi: qi + batch_q]
        sims    = q_batch @ c_vecs.T
        top_idx = np.argpartition(-sims, kth=min(top_k, sims.shape[1] - 1), axis=1)[:, :top_k]
        for bi in range(q_batch.shape[0]):
            qid = q_ids[qi + bi]
            relevant = qrels.get(qid)
            if not relevant:
                continue
            row_sims  = sims[bi, top_idx[bi]]
            order     = np.argsort(-row_sims)
            ranked    = c_ids_arr[top_idx[bi][order]]
            ndcg_scores.append(ndcg_at_k(ranked, relevant, 10))
            recall_scores.append(recall_at_k(ranked, relevant, 100))
    elapsed = time.perf_counter() - t0
    return EvalResult(
        label=label, group=group, bits_per_vec=bits_per_vec,
        ndcg10=float(np.mean(ndcg_scores))    if ndcg_scores    else 0.0,
        recall100=float(np.mean(recall_scores)) if recall_scores else 0.0,
        score_ms=elapsed * 1000,
    )


# ---------------------------------------------------------------------------
# Coordinate conversion  (float32 throughout to halve memory vs float64)
# ---------------------------------------------------------------------------

def to_angles(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    N, n  = x.shape
    out   = np.zeros((N, n - 1), dtype=np.float32)
    sq    = x ** 2
    tail  = np.sqrt(np.clip(np.cumsum(sq[:, ::-1], axis=1)[:, ::-1], 0.0, None))
    del sq
    for i in range(n - 2):
        out[:, i] = np.arccos(np.clip(x[:, i] / np.clip(tail[:, i], 1e-7, None), -1.0, 1.0))
    last = np.arctan2(x[:, n - 1], x[:, n - 2])
    out[:, n - 2] = np.where(last < 0, last + 2 * math.pi, last)
    del tail, last
    return out


def from_angles(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    N, m  = a.shape
    x     = np.zeros((N, m + 1), dtype=np.float32)
    sp    = np.ones(N, dtype=np.float32)
    for i in range(m):
        x[:, i] = sp * np.cos(a[:, i])
        sp = sp * np.sin(a[:, i])
    x[:, m] = sp
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-7, None)


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def quant_cartesian(vecs: np.ndarray, bits: int) -> np.ndarray:
    """Uniform scalar quantization of Cartesian unit vectors."""
    levels = 1 << bits
    scale  = (levels - 1) / 2.0
    q = np.clip(np.round(np.clip(vecs, -1.0, 1.0) * scale + scale / 2), 0, levels - 1).astype(np.int32)
    out = (q.astype(np.float32) + 0.5) / scale - 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-7, None)


def quant_angles_uniform(angles: np.ndarray, bits: int,
                          hi: np.ndarray) -> np.ndarray:
    """Uniform scalar quantization of all angles at `bits` bits each."""
    levels = 1 << bits
    a = np.clip(angles, 0.0, hi - 1e-6)
    q = np.clip(np.floor(a / hi * levels).astype(np.int32), 0, levels - 1)
    return (q.astype(np.float32) + 0.5) * (hi / levels)


def quant_angles_truncation(angles: np.ndarray, hi: np.ndarray,
                             keep_bits: int, keep_n: int,
                             clip_means: np.ndarray) -> np.ndarray:
    """
    Truncation scheme: keep the first `keep_n` angles at `keep_bits` bits,
    clip the rest to their corpus mean (0-bit).

    This is the core of the proposal: early angles (high Jacobian weight)
    are kept at full precision; late angles (low sensitivity) are dropped.
    """
    out = clip_means[None, :].repeat(angles.shape[0], axis=0).astype(np.float32)
    if keep_n > 0:
        out[:, :keep_n] = quant_angles_uniform(angles[:, :keep_n],
                                                keep_bits, hi[:keep_n])
    return out


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_all(
    c_vecs: np.ndarray, q_vecs: np.ndarray,
    c_ids: list[str], q_ids: list[str],
    qrels: dict[str, dict[str, int]],
    batch_q: int,
) -> tuple[list[EvalResult], list[EvalResult]]:
    dim      = c_vecs.shape[1]
    n_angles = dim - 1
    results:  list[EvalResult] = []
    clipping: list[EvalResult] = []   # separate: used for clipping plot

    def ev(label, group, bits, cv, qv):
        print(f"  {label} ...")
        r = evaluate(label, group, bits, qv, cv, q_ids, c_ids, qrels, batch_q)
        print(f"    NDCG@10={r.ndcg10:.4f}  R@100={r.recall100:.4f}  {r.score_ms/1000:.1f}s")
        return r

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    results.append(ev("float32 baseline", "baseline", 32 * dim, c_vecs, q_vecs))

    # -----------------------------------------------------------------------
    # TODO 1 & 2 -- Cartesian int vs Angle int at matched bit budgets
    # -----------------------------------------------------------------------
    print("\n--- Cartesian quantization ---")
    for bits in (8, 4, 2):
        cv = quant_cartesian(c_vecs, bits)
        qv = quant_cartesian(q_vecs, bits)
        results.append(ev(f"Cartesian int{bits}",
                          "cartesian", bits * dim, cv, qv))
        del cv, qv; gc.collect()

    print("\n--- Angle uniform quantization ---")
    hi = np.full(n_angles, math.pi, dtype=np.float32)
    hi[-1] = 2 * math.pi
    print("  converting to angles ...")
    c_angles = to_angles(c_vecs)
    q_angles = to_angles(q_vecs)
    clip_means = c_angles.mean(axis=0)
    gc.collect()

    for bits in (8, 4, 2):
        deq_c = quant_angles_uniform(c_angles, bits, hi)
        deq_q = quant_angles_uniform(q_angles, bits, hi)
        cv = from_angles(deq_c); del deq_c
        qv = from_angles(deq_q); del deq_q
        results.append(ev(f"Angle int{bits} (uniform)",
                          "angle-uniform", bits * n_angles, cv, qv))
        del cv, qv; gc.collect()

    # -----------------------------------------------------------------------
    # TODO 3 -- Clipping sweep
    # Fraction of angles clipped (last k% replaced by corpus mean).
    # Remaining angles kept at int8.
    # -----------------------------------------------------------------------
    print("\n--- Clipping sweep (remaining angles at int8) ---")
    clip_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for frac in clip_fractions:
        keep_n = int(round(n_angles * (1.0 - frac)))
        # bits stored = keep_n * 8; rest is free (corpus mean shared)
        stored_bits = keep_n * 8
        deq_c = quant_angles_truncation(c_angles, hi, 8, keep_n, clip_means)
        deq_q = quant_angles_truncation(q_angles, hi, 8, keep_n, clip_means)
        cv = from_angles(deq_c); del deq_c
        qv = from_angles(deq_q); del deq_q
        label = (f"Clip {int(frac*100):3d}% angles  "
                 f"(keep {keep_n}/{n_angles} @ int8 = {stored_bits} bits)")
        r = ev(label, "clipping", stored_bits, cv, qv)
        clipping.append(r)
        results.append(r)
        del cv, qv; gc.collect()

    # -----------------------------------------------------------------------
    # TODO 4 -- Truncation tradeoff
    # Vary how many angles are kept (at int8) vs clipped; also sweep int4/int2
    # for the kept angles to show redundancy of sub-int8 precision.
    # -----------------------------------------------------------------------
    print("\n--- Truncation tradeoff sweep ---")
    keep_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for keep_bits in (8, 4, 2):
        for kf in keep_fracs:
            keep_n      = int(round(n_angles * kf))
            stored_bits = keep_n * keep_bits
            deq_c = quant_angles_truncation(c_angles, hi, keep_bits, keep_n, clip_means)
            deq_q = quant_angles_truncation(q_angles, hi, keep_bits, keep_n, clip_means)
            cv = from_angles(deq_c); del deq_c
            qv = from_angles(deq_q); del deq_q
            label = f"Truncation int{keep_bits} keep={int(kf*100)}%"
            group = f"truncation-int{keep_bits}"
            results.append(ev(label, group, stored_bits, cv, qv))
            del cv, qv; gc.collect()

    del c_angles, q_angles; gc.collect()
    return results, clipping


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_table(results: list[EvalResult], path: Path) -> None:
    base_bits = next((r.bits_per_vec for r in results if "float32" in r.label), 1)
    header = (f"{'method':<65} {'bits/vec':>10} {'ratio':>7} "
              f"{'NDCG@10':>9} {'R@100':>8} {'score_s':>9}")
    lines = [header, "-" * len(header)]
    for r in results:
        ratio = base_bits / r.bits_per_vec if r.bits_per_vec > 0 else float("inf")
        lines.append(
            f"{r.label:<65} {r.bits_per_vec:>10d} {ratio:>7.1f}x "
            f"{r.ndcg10:>9.4f} {r.recall100:>8.4f} {r.score_ms/1000:>9.2f}")
    text = "\n".join(lines)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"\ntable -> {path}")


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
    print(f"CSV  -> {path}")


def plot_main(results: list[EvalResult], path: Path) -> None:
    """TODO 1 & 2: Cartesian int vs Angle uniform at int8/int4/int2."""
    base  = next(r for r in results if "float32" in r.label)
    base_bits = base.bits_per_vec

    cart  = [r for r in results if r.group == "cartesian"]
    angle = [r for r in results if r.group == "angle-uniform"]

    fig, ax = plt.subplots(figsize=(8, 5))

    def ratio(r): return base_bits / r.bits_per_vec

    ax.scatter([ratio(r) for r in cart],  [r.ndcg10 for r in cart],
               color="#e74c3c", s=100, zorder=5, label="Cartesian int (8/4/2-bit)")
    for r in cart:
        ax.annotate(f"int{r.bits_per_vec // base.bits_per_vec * 32}",
                    (ratio(r), r.ndcg10), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="#e74c3c")

    ax.scatter([ratio(r) for r in angle], [r.ndcg10 for r in angle],
               color="#2ecc71", s=100, marker="^", zorder=5,
               label="Angle int (8/4/2-bit, uniform)")
    for r in angle:
        bits_per_angle = r.bits_per_vec // (base_bits // 32 - 1)
        ax.annotate(f"int{bits_per_angle}",
                    (ratio(r), r.ndcg10), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="#2ecc71")

    ax.axhline(base.ndcg10, color="black", linestyle="--",
               linewidth=1.2, label="float32 baseline")
    ax.set_xlabel("Compression ratio (vs float32)", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title("TODO 1 & 2: Cartesian int vs Angle int at matched budgets\n"
                 "(higher = better; same x means same storage cost)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot  -> {path}")


def plot_clipping(clipping: list[EvalResult], base_ndcg: float, path: Path) -> None:
    """TODO 3: NDCG@10 vs fraction of angles clipped (0-bit)."""
    # clipping list is ordered by frac 0.0 → 1.0
    fracs = [1.0 - (r.bits_per_vec / clipping[0].bits_per_vec)
             for r in clipping]
    # recompute from label
    fracs = []
    for r in clipping:
        import re
        m = re.search(r"Clip\s+(\d+)%", r.label)
        fracs.append(int(m.group(1)) / 100.0 if m else 0.0)

    ndcgs = [r.ndcg10 for r in clipping]
    bits  = [r.bits_per_vec for r in clipping]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fracs, ndcgs, "o-", color="#3498db", linewidth=2, markersize=8)
    ax1.axhline(base_ndcg, color="black", linestyle="--", linewidth=1.2,
                label="float32 baseline")
    ax1.set_xlabel("Fraction of angles clipped (replaced by corpus mean)", fontsize=11)
    ax1.set_ylabel("NDCG@10", fontsize=11)
    ax1.set_title("TODO 3: Clipping validation\n"
                  "(how much can we clip before quality drops?)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(bits, ndcgs, "o-", color="#9b59b6", linewidth=2, markersize=8)
    ax2.axhline(base_ndcg, color="black", linestyle="--", linewidth=1.2,
                label="float32 baseline")
    ax2.set_xlabel("Bits stored per vector (remaining int8 angles only)", fontsize=11)
    ax2.set_ylabel("NDCG@10", fontsize=11)
    ax2.set_title("TODO 3: Bits stored vs quality", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Clipping ablation — keeping first k% angles at int8, rest at 0-bit",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot  -> {path}")


def plot_tradeoff(results: list[EvalResult], path: Path) -> None:
    """TODO 4: Compression ratio vs NDCG@10 for all schemes."""
    base_bits = next((r.bits_per_vec for r in results if "float32" in r.label), 1)
    base_ndcg = next((r.ndcg10       for r in results if "float32" in r.label), 1.0)

    groups = {
        "cartesian":         ("Cartesian int (8/4/2)", "o",  "#e74c3c"),
        "angle-uniform":     ("Angle int uniform",     "^",  "#2ecc71"),
        "truncation-int8":   ("Truncation int8",       "s",  "#3498db"),
        "truncation-int4":   ("Truncation int4",       "D",  "#9b59b6"),
        "truncation-int2":   ("Truncation int2",       "P",  "#f39c12"),
        "clipping":          ("Clipping sweep",        "v",  "#95a5a6"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for group, (name, marker, color) in groups.items():
        pts = sorted(
            [(base_bits / r.bits_per_vec, r.ndcg10)
             for r in results if r.group == group and r.bits_per_vec > 0],
            key=lambda x: x[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        for ax in (ax1, ax2):
            ax.plot(xs, ys, marker=marker, label=name, color=color,
                    linewidth=1.5, markersize=7)

    for ax in (ax1, ax2):
        ax.axhline(base_ndcg, color="black", linestyle="--",
                   linewidth=1.2, label="float32 baseline")
        ax.set_xlabel("Compression ratio (vs float32)", fontsize=12)
        ax.set_ylabel("NDCG@10", fontsize=12)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    ax1.set_title("TODO 4: Compression vs NDCG@10 (linear)", fontsize=12)
    ax2.set_title("TODO 4: Compression vs NDCG@10 (log x)", fontsize=12)
    ax2.set_xscale("log")

    fig.suptitle("Embedding → Angles → Quantize: full tradeoff",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir",  required=True)
    ap.add_argument("--data-dir",   required=True)
    ap.add_argument("--embed-tag",  default="")
    ap.add_argument("--max-corpus", type=int, default=50_000,
                    help="Max corpus docs (default 50k, safe for 4 GB RAM). 0 = no limit.")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--output-dir", default="eval_results")
    ap.add_argument("--batch-q",    type=int, default=256)
    args = ap.parse_args()

    embed_dir = Path(args.embed_dir).expanduser().resolve()
    data_dir  = Path(args.data_dir).expanduser().resolve()
    out_dir   = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_corpus  = args.max_corpus  or None
    max_queries = args.max_queries or None

    print(f"Loading corpus  from {embed_dir / 'corpus'} ...")
    c_vecs, c_ids = load_split(embed_dir / "corpus",  args.embed_tag, max_corpus)
    print(f"  {c_vecs.shape[0]:,} docs  dim={c_vecs.shape[1]}")

    print(f"Loading queries from {embed_dir / 'queries'} ...")
    q_vecs, q_ids = load_split(embed_dir / "queries", args.embed_tag, max_queries)
    print(f"  {q_vecs.shape[0]:,} queries")

    print(f"Loading qrels from {data_dir} ...")
    all_qrels = load_qrels(data_dir)

    c_id_set = set(c_ids)
    filt_qids = [qid for qid in q_ids
                 if qid in all_qrels and
                 any(cid in c_id_set for cid in all_qrels[qid])]
    print(f"  {len(filt_qids):,} / {len(q_ids):,} queries have relevant docs in corpus sample")
    if not filt_qids:
        print("ERROR: no queries with relevant docs in corpus sample. Increase --max-corpus.")
        return

    qid_to_idx = {qid: i for i, qid in enumerate(q_ids)}
    q_vecs_f   = q_vecs[[qid_to_idx[qid] for qid in filt_qids]]
    qrels_f    = {qid: all_qrels[qid] for qid in filt_qids}

    print(f"\nEvaluating {len(filt_qids):,} queries x {len(c_ids):,} corpus docs")
    print(f"float32 baseline = {32 * c_vecs.shape[1]:,} bits/vec\n")

    results, clipping = run_all(c_vecs, q_vecs_f, c_ids, filt_qids, qrels_f, args.batch_q)

    base_ndcg = next(r.ndcg10 for r in results if "float32" in r.label)

    write_table(results, out_dir / "results_table.txt")
    write_csv(results,   out_dir / "results.csv")
    plot_main(results,   out_dir / "plot_main.png")
    if clipping:
        plot_clipping(clipping, base_ndcg, out_dir / "plot_clipping.png")
    plot_tradeoff(results, out_dir / "plot_tradeoff.png")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
