"""
Embed an entire BEIR-style dataset directory with a local SentenceTransformer
model (default: sentence-transformers/all-MiniLM-L6-v2, 384-dim).

Mirrors the layout / resumability / logging of embed_beir.py. The only
differences are:
  - No API. Model runs locally on CPU or GPU (auto-detected).
  - Output dir defaults to ./embeddings_minilm so it does not collide with the
    OpenAI run.
  - Chunk filenames embed the model name suffix to make accidental mixing
    obvious (e.g. chunk_0000000.minilm-l6.npy).

For each dataset it produces:
    <out-dir>/<dataset>/corpus/chunk_XXXXXXX.<tag>.npy       (float32, L2-normalized, dim=384)
    <out-dir>/<dataset>/corpus/chunk_XXXXXXX.<tag>.ids.txt
    <out-dir>/<dataset>/corpus/DONE
    <out-dir>/<dataset>/queries/...                          (same layout)

Resume rules and atomic write semantics are identical to embed_beir.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TAG   = "minilm-l6"      # appended to chunk filenames
DEFAULT_DIM   = 384
DEFAULT_BATCH_CPU = 32
DEFAULT_BATCH_GPU = 256
CHUNK_BATCHES = 40               # batches per flushed chunk file


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class BufferedFormatter(logging.Formatter):
    def format(self, record):
        return f"[{self.formatTime(record, '%H:%M:%S')}] {record.levelname:<5} {record.getMessage()}"


def build_logger(verbose: bool) -> tuple[logging.Logger, list[str]]:
    log_buffer: list[str] = []

    class BufferHandler(logging.Handler):
        def emit(self, record):
            log_buffer.append(self.format(record))

    logger = logging.getLogger("embed_beir_minilm")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = BufferedFormatter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    buf = BufferHandler()
    buf.setFormatter(fmt)
    logger.addHandler(buf)
    return logger, log_buffer


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

@dataclass
class Item:
    ident: str
    text: str


def _compose_corpus_text(rec: dict) -> str:
    title = rec.get("title") or ""
    text  = rec.get("text")  or ""
    if title and text:
        return f"{title}\n{text}"
    return title or text


def iter_corpus(path: Path) -> Iterator[Item]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yield Item(str(rec["_id"]), _compose_corpus_text(rec))


def iter_queries(path: Path) -> Iterator[Item]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yield Item(str(rec["_id"]), rec.get("text") or "")


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Chunk layout + resume
# ---------------------------------------------------------------------------

def chunk_path(out_dir: Path, idx: int, tag: str) -> Path:
    return out_dir / f"chunk_{idx:07d}.{tag}.npy"


def chunk_ids_path(out_dir: Path, idx: int, tag: str) -> Path:
    return out_dir / f"chunk_{idx:07d}.{tag}.ids.txt"


def scan_existing(out_dir: Path, tag: str, logger: logging.Logger) -> tuple[int, int]:
    if not out_dir.exists():
        return 0, 0
    chunks = sorted(out_dir.glob(f"chunk_*.{tag}.npy"))
    processed = 0
    good = 0
    for p in chunks:
        # filename: chunk_0000000.<tag>.npy  ->  ids = chunk_0000000.<tag>.ids.txt
        stem_no_npy = p.name[:-len(".npy")]
        ids_p = out_dir / (stem_no_npy + ".ids.txt")
        if not ids_p.exists():
            logger.warning(f"dropping orphan chunk {p.name}: no ids file")
            p.unlink()
            continue
        try:
            arr = np.load(p, mmap_mode="r")
            n_ids = sum(1 for _ in ids_p.open("r", encoding="utf-8"))
        except Exception as e:
            logger.warning(f"dropping unreadable chunk {p.name}: {e}")
            p.unlink()
            ids_p.unlink(missing_ok=True)
            continue
        if arr.shape[0] != n_ids:
            logger.warning(f"dropping mismatched chunk {p.name}: {arr.shape[0]} vecs vs {n_ids} ids")
            p.unlink()
            ids_p.unlink(missing_ok=True)
            continue
        processed += int(arr.shape[0])
        good += 1
    return good, processed


def flush_chunk(out_dir: Path, chunk_idx: int, tag: str,
                vecs: list[np.ndarray], ids: list[str]) -> None:
    if not vecs:
        return
    arr = np.concatenate(vecs, axis=0).astype(np.float32, copy=False)
    npy_p = chunk_path(out_dir, chunk_idx, tag)
    ids_p = chunk_ids_path(out_dir, chunk_idx, tag)
    tmp_npy = out_dir / (npy_p.name + ".tmp")
    tmp_ids = out_dir / (ids_p.name + ".tmp")
    with tmp_npy.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    with tmp_ids.open("w", encoding="utf-8") as f:
        for i in ids:
            f.write(i + "\n")
    os.replace(tmp_npy, npy_p)
    os.replace(tmp_ids, ids_p)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def make_encoder(model_name: str, device: str | None, fp16: bool,
                 logger: logging.Logger):
    # Imported lazily so a missing torch install fails with a clear message
    # only when this script is actually run.
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"loading model: {model_name}  (device={device})")
    model = SentenceTransformer(model_name, device=device)
    if fp16 and device.startswith("cuda"):
        model = model.half()
        logger.info("using fp16 on GPU")
    elif fp16 and not device.startswith("cuda"):
        logger.warning("fp16 requested but no CUDA device; staying in fp32")
    return model, device


def embed_batch(model, texts: list[str]) -> np.ndarray:
    # normalize_embeddings=True -> L2-normalized output, ready for cosine.
    vecs = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs.astype(np.float32, copy=False)


def embed_split(
    model,
    tag: str,
    split_name: str,
    items_iter_factory,
    total_items: int,
    out_dir: Path,
    batch_size: int,
    chunk_batches: int,
    logger: logging.Logger,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "DONE"
    stats = {"split": split_name, "total": total_items, "skipped": 0,
             "embedded": 0, "batches": 0, "seconds": 0.0}

    if done_marker.exists():
        logger.info(f"  {split_name}: already complete (DONE marker). skipping.")
        stats["skipped"] = total_items
        return stats

    n_chunks_done, already = scan_existing(out_dir, tag, logger)
    stats["skipped"] = already
    if already:
        logger.info(f"  {split_name}: resuming after {already} items in {n_chunks_done} chunks")

    chunk_idx = n_chunks_done
    chunk_size_items = batch_size * chunk_batches

    buf_vecs: list[np.ndarray] = []
    buf_ids:  list[str] = []
    batch_texts: list[str] = []
    batch_ids:   list[str] = []

    t0 = time.perf_counter()
    pbar = tqdm(total=total_items, initial=already, desc=f"{split_name:<7}",
                unit="doc", dynamic_ncols=True)
    seen = 0

    def flush_if_chunk_full():
        nonlocal chunk_idx, buf_vecs, buf_ids
        current = sum(v.shape[0] for v in buf_vecs)
        if current >= chunk_size_items:
            flush_chunk(out_dir, chunk_idx, tag, buf_vecs, buf_ids)
            logger.debug(f"  {split_name}: flushed chunk {chunk_idx} ({current} items)")
            chunk_idx += 1
            buf_vecs = []
            buf_ids = []

    def flush_batch():
        nonlocal batch_texts, batch_ids
        if not batch_texts:
            return
        vecs = embed_batch(model, batch_texts)
        buf_vecs.append(vecs)
        buf_ids.extend(batch_ids)
        stats["batches"] += 1
        stats["embedded"] += len(batch_texts)
        pbar.update(len(batch_texts))
        batch_texts = []
        batch_ids = []
        flush_if_chunk_full()

    try:
        for item in items_iter_factory():
            seen += 1
            if seen <= already:
                continue
            batch_ids.append(item.ident)
            batch_texts.append(item.text if item.text else " ")
            if len(batch_texts) >= batch_size:
                flush_batch()
        flush_batch()
        if buf_vecs:
            flush_chunk(out_dir, chunk_idx, tag, buf_vecs, buf_ids)
            logger.debug(f"  {split_name}: flushed final chunk {chunk_idx}")
        done_marker.write_text("ok\n")
    finally:
        pbar.close()

    stats["seconds"] = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_datasets(data_dir: Path, filter_names: list[str] | None) -> list[Path]:
    out = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if filter_names and child.name not in filter_names:
            continue
        if (child / "corpus.jsonl").exists() or (child / "queries.jsonl").exists():
            out.append(child)
        else:
            nested = [g for g in child.iterdir()
                      if g.is_dir() and (g / "corpus.jsonl").exists()]
            out.extend(nested)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default=os.environ.get("BEIR_DATA_DIR", "bier-data"))
    ap.add_argument("--output-dir", default=os.environ.get("EMBED_OUT_DIR_MINILM", "embeddings_minilm"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag",   default=DEFAULT_TAG,
                    help="Suffix tagged into chunk filenames; defaults to minilm-l6.")
    ap.add_argument("--device", default=None,
                    help='"cuda", "cpu", or omit to auto-detect.')
    ap.add_argument("--fp16", action="store_true",
                    help="Use half precision on GPU (no effect on CPU).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Default {DEFAULT_BATCH_GPU} on GPU, {DEFAULT_BATCH_CPU} on CPU.")
    ap.add_argument("--chunk-batches", type=int, default=CHUNK_BATCHES)
    ap.add_argument("--skip-corpus",  action="store_true")
    ap.add_argument("--skip-queries", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logger, log_buffer = build_logger(verbose=not args.quiet)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir  = Path(args.output_dir).expanduser().resolve()
    if not data_dir.exists():
        logger.error(f"data dir not found: {data_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(data_dir, args.datasets)
    if not datasets:
        logger.error(f"no BEIR-style datasets found under {data_dir}")
        sys.exit(1)

    model, device = make_encoder(args.model, args.device, args.fp16, logger)
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_GPU if device.startswith("cuda") else DEFAULT_BATCH_CPU

    logger.info(f"data_dir   = {data_dir}")
    logger.info(f"output_dir = {out_dir}")
    logger.info(f"model      = {args.model}  (tag={args.tag})")
    logger.info(f"device     = {device}  fp16={args.fp16}")
    logger.info(f"batch_size = {args.batch_size}  chunk_batches = {args.chunk_batches}")
    logger.info(f"datasets   = {[d.name for d in datasets]}")

    run_start = time.perf_counter()
    all_stats: list[dict] = []

    for ds_path in datasets:
        ds_name = ds_path.name
        ds_out  = out_dir / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n=== dataset: {ds_name} ===")

        corpus_file  = ds_path / "corpus.jsonl"
        queries_file = ds_path / "queries.jsonl"

        if not args.skip_corpus and corpus_file.exists():
            logger.info(f"  counting lines in {corpus_file.name} ...")
            n = count_lines(corpus_file)
            logger.info(f"  corpus: {n} docs")
            s = embed_split(
                model, args.tag, "corpus",
                lambda: iter_corpus(corpus_file),
                n, ds_out / "corpus",
                args.batch_size, args.chunk_batches, logger,
            )
            s["dataset"] = ds_name
            all_stats.append(s)
        elif not corpus_file.exists():
            logger.warning(f"  no corpus.jsonl in {ds_path}")

        if not args.skip_queries and queries_file.exists():
            logger.info(f"  counting lines in {queries_file.name} ...")
            n = count_lines(queries_file)
            logger.info(f"  queries: {n} queries")
            s = embed_split(
                model, args.tag, "queries",
                lambda: iter_queries(queries_file),
                n, ds_out / "queries",
                args.batch_size, args.chunk_batches, logger,
            )
            s["dataset"] = ds_name
            all_stats.append(s)
        elif not queries_file.exists():
            logger.warning(f"  no queries.jsonl in {ds_path}")

    total_secs = time.perf_counter() - run_start

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'dataset':<25} {'split':<8} {'total':>10} {'skipped':>10} {'embedded':>10} {'batches':>8} {'sec':>8}")
    for s in all_stats:
        logger.info(f"{s['dataset']:<25} {s['split']:<8} {s['total']:>10} {s['skipped']:>10} {s['embedded']:>10} {s['batches']:>8} {s['seconds']:>8.1f}")
    logger.info("-" * 80)
    logger.info(f"total wall clock: {total_secs:.1f} s")

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"embed_beir_minilm_log_{ts}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(log_buffer) + "\n")
    print(f"\nrun log written to: {log_path}")


if __name__ == "__main__":
    main()
