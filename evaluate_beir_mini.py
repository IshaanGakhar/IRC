"""
Cross-subset evaluation driver.

For each BEIR subset present under --embed-root and --data-root, invokes
`evaluate_streaming.py` via subprocess, then produces a single aggregated
table with NDCG@{5,10,15,20} and R@100 averaged across subsets.

Input layout (must both match):
    --embed-root/<subset>/corpus/...    .npy chunks (from embed_beir.py)
    --embed-root/<subset>/queries/...
    --data-root/<subset>/corpus.jsonl
    --data-root/<subset>/queries.jsonl
    --data-root/<subset>/qrels/test.tsv

Output:
    --output-root/<subset>/results_streaming.{txt,csv}
    --output-root/_aggregate_by_scheme.{txt,csv}        per scheme, macro-avg
    --output-root/_aggregate_by_subset.{txt,csv}        per subset, all schemes

Usage:
    python evaluate_beir_mini.py \\
        --embed-root ./embeddings_oai_mini \\
        --data-root  ~/bier-data-mini \\
        --output-root ./eval_results_mini \\
        --embed-tag "" \\
        --super-batch 50000 --q-batch 1024
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _discover_subsets(embed_root: Path, data_root: Path) -> list[str]:
    """Only return subsets that exist in both roots and have corpus chunks."""
    out = []
    for p in sorted(embed_root.iterdir()):
        if not p.is_dir():
            continue
        corpus_dir = p / "corpus"
        if not corpus_dir.exists():
            continue
        if not any(corpus_dir.glob("chunk_*.npy")):
            continue
        if not (data_root / p.name).exists():
            continue
        out.append(p.name)
    return out


def run_one_subset(
    subset: str,
    embed_root: Path,
    data_root: Path,
    output_root: Path,
    embed_tag: str,
    super_batch: int,
    q_batch: int,
    checkpoint_every: int,
    scheme_workers: int,
) -> list[dict]:
    """Shell out to evaluate_streaming.py; return list of result rows."""
    out_dir = output_root / subset
    cmd = [
        sys.executable, "evaluate_streaming.py",
        "--embed-dir",  str(embed_root / subset),
        "--data-dir",   str(data_root / subset),
        "--embed-tag",  embed_tag,
        "--output-dir", str(out_dir),
        "--super-batch", str(super_batch),
        "--q-batch",    str(q_batch),
        "--checkpoint-every", str(checkpoint_every),
    ]
    if scheme_workers > 0:
        cmd += ["--scheme-workers", str(scheme_workers)]

    print(f"\n{'='*72}\n[{subset}]  running evaluate_streaming.py\n{'='*72}",
          flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"[{subset}] evaluate_streaming.py exited "
                           f"{res.returncode}")

    csv_path = out_dir / "results_streaming.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} not produced.")
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["_subset"] = subset
            rows.append(r)
    return rows


def _num_cols(rows: list[dict]) -> list[str]:
    """Columns that look numeric across every row (ndcg*, recall*, etc.)."""
    cols = []
    for k in rows[0].keys():
        if k.startswith("ndcg") or k.startswith("recall"):
            try:
                float(rows[0][k])
                cols.append(k)
            except (ValueError, TypeError):
                pass
    return cols


def aggregate_by_scheme(all_rows: list[dict], output_root: Path) -> None:
    """Macro-average each metric across subsets, grouped by scheme."""
    by_scheme: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_scheme[r["scheme"]].append(r)
    metric_cols = _num_cols(all_rows)

    agg: list[dict] = []
    for scheme, rows in by_scheme.items():
        macro = {c: sum(float(r[c]) for r in rows) / len(rows) for c in metric_cols}
        bits = int(float(rows[0]["bits_per_vec"]))
        try:
            ratio = float(rows[0]["compression_ratio"])
        except (KeyError, ValueError):
            ratio = math.nan
        agg.append({
            "scheme":       scheme,
            "bits_per_vec": bits,
            "compression":  ratio,
            **macro,
            "n_subsets":    len(rows),
        })

    # sort by bits descending to match result-file convention
    agg.sort(key=lambda r: -r["bits_per_vec"])

    ndcg_cols = sorted([c for c in metric_cols if c.startswith("ndcg")],
                       key=lambda s: int(s[4:]))
    rec_cols  = sorted([c for c in metric_cols if c.startswith("recall")],
                       key=lambda s: int(s[6:]))
    cols_in_order = ndcg_cols + rec_cols

    # Text table
    hdr_parts = [f"{'scheme':<45}", f"{'bits':>8}", f"{'ratio':>7}"]
    for c in ndcg_cols:
        hdr_parts.append(f"{'NDCG@'+c[4:]:>9}")
    for c in rec_cols:
        hdr_parts.append(f"{'R@'+c[6:]:>8}")
    hdr_parts.append(f"{'nsub':>5}")
    hdr = " ".join(hdr_parts)
    lines = [hdr, "-" * len(hdr)]
    for r in agg:
        parts = [f"{r['scheme']:<45}",
                 f"{r['bits_per_vec']:>8d}",
                 f"{r['compression']:>6.1f}x"]
        for c in ndcg_cols:
            parts.append(f"{r[c]:>9.4f}")
        for c in rec_cols:
            parts.append(f"{r[c]:>8.4f}")
        parts.append(f"{r['n_subsets']:>5d}")
        lines.append(" ".join(parts))
    text = "\n".join(lines) + "\n"
    txt_path = output_root / "_aggregate_by_scheme.txt"
    txt_path.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"aggregate by scheme -> {txt_path}")

    csv_path = output_root / "_aggregate_by_scheme.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f,
            fieldnames=["scheme", "bits_per_vec", "compression"]
                      + cols_in_order + ["n_subsets"])
        w.writeheader()
        for r in agg:
            w.writerow(r)
    print(f"                   -> {csv_path}")


def aggregate_by_subset(all_rows: list[dict], output_root: Path) -> None:
    """One table per subset, showing each scheme's score on that subset."""
    metric_cols = _num_cols(all_rows)
    ndcg_cols = sorted([c for c in metric_cols if c.startswith("ndcg")],
                       key=lambda s: int(s[4:]))
    rec_cols  = sorted([c for c in metric_cols if c.startswith("recall")],
                       key=lambda s: int(s[6:]))

    csv_path = output_root / "_aggregate_by_subset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "scheme", "bits_per_vec",
                                          "compression_ratio"]
                                         + ndcg_cols + rec_cols
                                         + ["n_queries", "n_corpus"])
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "subset":            r["_subset"],
                "scheme":            r["scheme"],
                "bits_per_vec":      int(float(r["bits_per_vec"])),
                "compression_ratio": r.get("compression_ratio", ""),
                **{c: r[c] for c in ndcg_cols + rec_cols},
                "n_queries":         r.get("n_queries", ""),
                "n_corpus":          r.get("n_corpus", ""),
            })
    print(f"per-subset rows  -> {csv_path}")

    # A compact float32-baseline comparison table per subset
    by_subset: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in all_rows:
        by_subset[r["_subset"]][r["scheme"]] = r
    lines = ["Baseline (float32) NDCG@10 + R@100 per subset:", ""]
    lines.append(f"{'subset':<18}{'NDCG@5':>9}{'NDCG@10':>10}"
                 f"{'NDCG@15':>10}{'NDCG@20':>10}{'R@100':>9}"
                 f"{'queries':>10}{'corpus':>10}")
    lines.append("-" * 86)
    for subset, schemes in sorted(by_subset.items()):
        f32 = schemes.get("float32")
        if not f32:
            continue
        lines.append(
            f"{subset:<18}"
            f"{float(f32.get('ndcg5', 0)):>9.4f}"
            f"{float(f32.get('ndcg10', 0)):>10.4f}"
            f"{float(f32.get('ndcg15', 0)):>10.4f}"
            f"{float(f32.get('ndcg20', 0)):>10.4f}"
            f"{float(f32.get('recall100', 0)):>9.4f}"
            f"{int(float(f32.get('n_queries', 0))):>10,}"
            f"{int(float(f32.get('n_corpus', 0))):>10,}"
        )
    baseline_text = "\n".join(lines) + "\n"
    bp = output_root / "_baselines_by_subset.txt"
    bp.write_text(baseline_text, encoding="utf-8")
    print("\n" + baseline_text)
    print(f"baselines -> {bp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", required=True,
                    help="Root containing <subset>/corpus and <subset>/queries.")
    ap.add_argument("--data-root",  required=True,
                    help="Root containing <subset>/{corpus,queries,qrels}.")
    ap.add_argument("--output-root", default="eval_results_mini")
    ap.add_argument("--subsets",    nargs="*", default=None,
                    help="Which subsets to evaluate (default: all discovered).")
    ap.add_argument("--embed-tag",  default="",
                    help='Chunk filename tag ("" for plain OAI layout).')
    ap.add_argument("--super-batch",      type=int, default=50_000)
    ap.add_argument("--q-batch",          type=int, default=1024)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--scheme-workers",   type=int, default=0,
                    help="0 = evaluate_streaming.py default (auto).")
    args = ap.parse_args()

    embed_root = Path(args.embed_root).expanduser().resolve()
    data_root  = Path(args.data_root).expanduser().resolve()
    out_root   = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    discovered = _discover_subsets(embed_root, data_root)
    if args.subsets:
        subsets = [s for s in args.subsets if s in discovered]
        missing = [s for s in args.subsets if s not in discovered]
        if missing:
            print(f"  [warn] requested subsets not found: {missing}")
    else:
        subsets = discovered
    if not subsets:
        print("ERROR: no subsets found to evaluate.")
        sys.exit(1)

    print(f"embed_root = {embed_root}")
    print(f"data_root  = {data_root}")
    print(f"output     = {out_root}")
    print(f"subsets    = {subsets}\n")

    all_rows: list[dict] = []
    for subset in subsets:
        rows = run_one_subset(
            subset           = subset,
            embed_root       = embed_root,
            data_root        = data_root,
            output_root      = out_root,
            embed_tag        = args.embed_tag,
            super_batch      = args.super_batch,
            q_batch          = args.q_batch,
            checkpoint_every = args.checkpoint_every,
            scheme_workers   = args.scheme_workers,
        )
        all_rows.extend(rows)

    # Aggregates across subsets
    aggregate_by_subset(all_rows, out_root)
    aggregate_by_scheme(all_rows, out_root)

    print(f"\nAll done. Results under: {out_root}")


if __name__ == "__main__":
    main()
