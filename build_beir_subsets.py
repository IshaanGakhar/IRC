"""
Build compact BEIR-style subsets for cross-dataset evaluation.

For each of the 8 "standard" BEIR subsets, samples min(1000, all_test_queries)
queries that have qrels, collects *every* relevant corpus doc for those queries,
and adds a fixed number of random distractor docs. Writes the result in the
standard BEIR layout so `embed_beir.py` / `evaluate_streaming.py` can run on it
without modification.

Input layout (what you need in --src-root):
    <src-root>/<subset>/corpus.jsonl
    <src-root>/<subset>/queries.jsonl
    <src-root>/<subset>/qrels/test.tsv        (or dev.tsv / train.tsv)

Output layout (what this script writes to --dst-root):
    <dst-root>/<subset>/corpus.jsonl          (rel docs + distractors)
    <dst-root>/<subset>/queries.jsonl         (sampled queries only)
    <dst-root>/<subset>/qrels/test.tsv        (qrels for sampled queries)
    <dst-root>/<subset>/_build_info.json      (stats: sizes, seed, etc.)

Usage:
    python build_beir_subsets.py \\
        --src-root ~/bier-data \\
        --dst-root ~/bier-data-mini \\
        --n-queries 1000 \\
        --n-distractors 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

STANDARD_SUBSETS = [
    "scifact",
    "nfcorpus",
    "fiqa",
    "arguana",
    "trec-covid",
    "nq",
    "quora",
    "hotpotqa",
]


def _find_qrels_file(qrels_dir: Path) -> Path:
    for name in ("test.tsv", "dev.tsv", "train.tsv"):
        p = qrels_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No qrels file in {qrels_dir}")


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                out[row["query-id"]][row["corpus-id"]] = score
    return out


def build_subset(
    subset_name: str,
    src_dir: Path,
    dst_dir: Path,
    n_queries: int,
    n_distractors: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)

    # Load qrels, determine sampled queries + their relevant docs
    qrels_src = _find_qrels_file(src_dir / "qrels")
    qrels = _load_qrels(qrels_src)
    qids_with_rel = sorted(qrels.keys())
    if len(qids_with_rel) > n_queries:
        sampled_qids = set(rng.sample(qids_with_rel, n_queries))
    else:
        sampled_qids = set(qids_with_rel)

    rel_doc_ids: set[str] = set()
    for qid in sampled_qids:
        rel_doc_ids.update(qrels[qid].keys())

    # Scan corpus once: collect all doc IDs so we can sample distractors
    # disjoint from relevant docs.
    corpus_src = src_dir / "corpus.jsonl"
    all_doc_ids: list[str] = []
    with corpus_src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            all_doc_ids.append(str(rec["_id"]))
    all_set = set(all_doc_ids)

    missing_rel = rel_doc_ids - all_set
    if missing_rel:
        print(f"  [{subset_name}] WARN: {len(missing_rel)} qrel doc IDs not "
              f"found in corpus.jsonl; they'll be dropped.")
        rel_doc_ids &= all_set

    # Sample distractors from the complement of rel_doc_ids
    distractor_pool = [d for d in all_doc_ids if d not in rel_doc_ids]
    n_distr = min(n_distractors, len(distractor_pool))
    distractor_ids = set(rng.sample(distractor_pool, n_distr))
    keep_doc_ids = rel_doc_ids | distractor_ids

    # Write corpus subset
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_corpus = dst_dir / "corpus.jsonl"
    n_written_docs = 0
    with corpus_src.open("r", encoding="utf-8") as fin, \
         dst_corpus.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec["_id"]) in keep_doc_ids:
                fout.write(line if line.endswith("\n") else line + "\n")
                n_written_docs += 1

    # Write queries subset
    queries_src = src_dir / "queries.jsonl"
    dst_queries = dst_dir / "queries.jsonl"
    n_written_queries = 0
    with queries_src.open("r", encoding="utf-8") as fin, \
         dst_queries.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec["_id"]) in sampled_qids:
                fout.write(line if line.endswith("\n") else line + "\n")
                n_written_queries += 1

    # Write qrels subset (only rows for kept query/doc combinations)
    dst_qrels = dst_dir / "qrels" / "test.tsv"
    dst_qrels.parent.mkdir(parents=True, exist_ok=True)
    n_qrels = 0
    with dst_qrels.open("w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid in sampled_qids:
            for cid, score in qrels[qid].items():
                if cid in keep_doc_ids:
                    f.write(f"{qid}\t{cid}\t{score}\n")
                    n_qrels += 1

    info = {
        "subset":           subset_name,
        "src_qrels_file":   qrels_src.name,
        "queries_written":  n_written_queries,
        "queries_requested": n_queries,
        "corpus_written":   n_written_docs,
        "rel_docs":         len(rel_doc_ids),
        "distractors":      n_distr,
        "qrels_rows":       n_qrels,
        "seed":             seed,
    }
    (dst_dir / "_build_info.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root",      required=True,
                    help="Dir containing each subset as a subdirectory in BEIR "
                         "layout (corpus.jsonl, queries.jsonl, qrels/).")
    ap.add_argument("--dst-root",      required=True,
                    help="Where to write the compact subsets.")
    ap.add_argument("--subsets", nargs="+", default=STANDARD_SUBSETS,
                    help=f"Subset names to build. Default: {STANDARD_SUBSETS}")
    ap.add_argument("--n-queries",     type=int, default=1000,
                    help="Max queries per subset (uses all if fewer exist).")
    ap.add_argument("--n-distractors", type=int, default=10_000,
                    help="Random distractor docs per subset.")
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"src_root = {src_root}")
    print(f"dst_root = {dst_root}")
    print(f"subsets  = {args.subsets}")
    print(f"sampling = {args.n_queries} queries + {args.n_distractors} distractors\n")

    all_info = []
    for name in args.subsets:
        src = src_root / name
        if not src.exists():
            print(f"  [{name}] SKIP: {src} does not exist")
            continue
        print(f"  [{name}] building ...", flush=True)
        info = build_subset(
            subset_name   = name,
            src_dir       = src,
            dst_dir       = dst_root / name,
            n_queries     = args.n_queries,
            n_distractors = args.n_distractors,
            seed          = args.seed,
        )
        all_info.append(info)
        print(f"  [{name}] -> {info['queries_written']:>5,} queries  "
              f"{info['corpus_written']:>7,} corpus docs  "
              f"({info['rel_docs']:>5,} rel + {info['distractors']:>5,} distr)  "
              f"{info['qrels_rows']:>6,} qrels rows")

    # Summary
    summary_path = dst_root / "_build_summary.json"
    summary_path.write_text(json.dumps(all_info, indent=2))
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
