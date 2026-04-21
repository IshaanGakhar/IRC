"""
Embed an entire BEIR-style dataset directory with OpenAI embeddings.

Assumes the standard BEIR layout, one subdirectory per dataset:
    <data-dir>/<dataset>/corpus.jsonl
    <data-dir>/<dataset>/queries.jsonl
    <data-dir>/<dataset>/qrels/...              (not embedded)

For each dataset it produces:
    <out-dir>/<dataset>/corpus/chunk_XXXXXXX.npy     (float32, L2-normalized)
    <out-dir>/<dataset>/corpus/chunk_XXXXXXX.ids.txt
    <out-dir>/<dataset>/corpus/DONE                  (written when split completes)
    <out-dir>/<dataset>/queries/...                  (same layout)

Resumability:
    - Each chunk is written atomically (tmp + rename) so a killed process cannot
      leave a half-written chunk.
    - On restart, existing chunks are counted, their ids are skipped, and work
      resumes from the next unseen line. Chunks are deterministic in size
      (except possibly the final chunk), so restart is exact.
    - Presence of `DONE` means that (dataset, split) is complete -- skipped.

Logging:
    - Verbose to stdout via a Logger.
    - Buffered log lines also written to a timestamped run log at the end.
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
from openai import OpenAI
from tqdm import tqdm

try:
    import tiktoken
except ImportError:
    tiktoken = None

load_dotenv()

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM   = 1536
DEFAULT_BATCH = 256          # items per API call
CHUNK_BATCHES = 40           # batches per flushed .npy chunk file
MAX_TOKENS    = 8000         # text-embedding-3-small hard cap is 8192


class BufferedFormatter(logging.Formatter):
    def format(self, record):
        return f"[{self.formatTime(record, '%H:%M:%S')}] {record.levelname:<5} {record.getMessage()}"


def build_logger(verbose: bool) -> tuple[logging.Logger, list[str]]:
    log_buffer: list[str] = []

    class BufferHandler(logging.Handler):
        def emit(self, record):
            log_buffer.append(self.format(record))

    logger = logging.getLogger("embed_beir")
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
# IO helpers
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
# Token-aware truncation
# ---------------------------------------------------------------------------

class Truncator:
    def __init__(self, model: str, max_tokens: int):
        self.max_tokens = max_tokens
        self.enc = None
        if tiktoken is not None:
            try:
                self.enc = tiktoken.encoding_for_model(model)
            except Exception:
                self.enc = tiktoken.get_encoding("cl100k_base")

    def truncate(self, text: str) -> str:
        if not text:
            return " "  # OpenAI rejects empty strings
        if self.enc is None:
            # char fallback -- roughly 4 chars/token
            limit = self.max_tokens * 4
            return text[:limit] if len(text) > limit else text
        toks = self.enc.encode(text)
        if len(toks) <= self.max_tokens:
            return text
        return self.enc.decode(toks[: self.max_tokens])


# ---------------------------------------------------------------------------
# Chunk layout + resume
# ---------------------------------------------------------------------------

def chunk_path(out_dir: Path, idx: int) -> Path:
    return out_dir / f"chunk_{idx:07d}.npy"


def chunk_ids_path(out_dir: Path, idx: int) -> Path:
    return out_dir / f"chunk_{idx:07d}.ids.txt"


def scan_existing(out_dir: Path, logger: logging.Logger) -> tuple[int, int]:
    """Return (num_chunks, num_items_already_embedded).

    A chunk counts only if BOTH the .npy and .ids.txt exist and are consistent.
    Any incomplete trailing pair is removed so the next write starts clean.
    """
    if not out_dir.exists():
        return 0, 0

    chunks = sorted(out_dir.glob("chunk_*.npy"))
    processed = 0
    good = 0
    for p in chunks:
        ids_p = out_dir / (p.stem + ".ids.txt")
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


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(client: OpenAI, model: str, batch: list[str],
                logger: logging.Logger, max_retries: int = 6) -> np.ndarray:
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=batch)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.clip(norms, 1e-12, None)
        except Exception as e:
            last_err = e
            logger.warning(f"embed error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}. sleep {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError(f"embedding failed after {max_retries} retries: {last_err}")


def flush_chunk(out_dir: Path, chunk_idx: int,
                vecs: list[np.ndarray], ids: list[str]) -> None:
    if not vecs:
        return
    arr = np.concatenate(vecs, axis=0).astype(np.float32, copy=False)
    npy_p = chunk_path(out_dir, chunk_idx)
    ids_p = chunk_ids_path(out_dir, chunk_idx)
    tmp_npy = out_dir / (npy_p.name + ".tmp")
    tmp_ids = out_dir / (ids_p.name + ".tmp")
    with tmp_npy.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    with tmp_ids.open("w", encoding="utf-8") as f:
        for i in ids:
            f.write(i + "\n")
    os.replace(tmp_npy, npy_p)
    os.replace(tmp_ids, ids_p)


def embed_split(
    client: OpenAI,
    model: str,
    split_name: str,
    items_iter_factory,                     # () -> Iterator[Item]
    total_items: int,
    out_dir: Path,
    truncator: Truncator,
    batch_size: int,
    chunk_batches: int,
    logger: logging.Logger,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "DONE"
    stats = {"split": split_name, "total": total_items, "skipped": 0, "embedded": 0,
             "batches": 0, "seconds": 0.0}

    if done_marker.exists():
        logger.info(f"  {split_name}: already complete (DONE marker). skipping.")
        stats["skipped"] = total_items
        return stats

    n_chunks_done, already = scan_existing(out_dir, logger)
    stats["skipped"] = already
    if already:
        logger.info(f"  {split_name}: resuming after {already} items in {n_chunks_done} chunks")

    chunk_idx = n_chunks_done
    chunk_size_items = batch_size * chunk_batches

    buf_vecs: list[np.ndarray] = []
    buf_ids:  list[str]        = []
    batch_texts: list[str] = []
    batch_ids:   list[str] = []

    t0 = time.perf_counter()
    seen = 0
    pbar = tqdm(total=total_items, initial=already, desc=f"{split_name:<7}",
                unit="doc", dynamic_ncols=True)

    def flush_if_chunk_full():
        nonlocal chunk_idx, buf_vecs, buf_ids
        current = sum(v.shape[0] for v in buf_vecs)
        if current >= chunk_size_items:
            flush_chunk(out_dir, chunk_idx, buf_vecs, buf_ids)
            logger.debug(f"  {split_name}: flushed chunk {chunk_idx} ({current} items)")
            chunk_idx += 1
            buf_vecs = []
            buf_ids  = []

    def flush_batch():
        nonlocal batch_texts, batch_ids
        if not batch_texts:
            return
        vecs = embed_batch(client, model, batch_texts, logger)
        buf_vecs.append(vecs)
        buf_ids.extend(batch_ids)
        stats["batches"] += 1
        stats["embedded"] += len(batch_texts)
        pbar.update(len(batch_texts))
        batch_texts = []
        batch_ids   = []
        flush_if_chunk_full()

    try:
        for item in items_iter_factory():
            seen += 1
            if seen <= already:
                continue
            batch_ids.append(item.ident)
            batch_texts.append(truncator.truncate(item.text))
            if len(batch_texts) >= batch_size:
                flush_batch()
        flush_batch()
        if buf_vecs:
            flush_chunk(out_dir, chunk_idx, buf_vecs, buf_ids)
            logger.debug(f"  {split_name}: flushed final chunk {chunk_idx}")
        done_marker.write_text("ok\n")
    finally:
        pbar.close()

    stats["seconds"] = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Top-level driver
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
            nested = [g for g in child.iterdir() if g.is_dir() and (g / "corpus.jsonl").exists()]
            out.extend(nested)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default=os.environ.get("BEIR_DATA_DIR", "bier-data"),
                    help="Root directory containing per-dataset subfolders (default: $BEIR_DATA_DIR or ./bier-data).")
    ap.add_argument("--output-dir", default=os.environ.get("EMBED_OUT_DIR", "embeddings"),
                    help="Where embeddings get written (default: ./embeddings).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Only process these dataset names (default: all).")
    ap.add_argument("--model",      default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--chunk-batches", type=int, default=CHUNK_BATCHES,
                    help="How many batches per flushed chunk file.")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--skip-corpus",  action="store_true")
    ap.add_argument("--skip-queries", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logger, log_buffer = build_logger(verbose=not args.quiet)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Put it in .env or export it.")
        sys.exit(1)

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

    logger.info(f"data_dir   = {data_dir}")
    logger.info(f"output_dir = {out_dir}")
    logger.info(f"model      = {args.model}")
    logger.info(f"batch_size = {args.batch_size}  chunk_batches = {args.chunk_batches}  max_tokens = {args.max_tokens}")
    logger.info(f"datasets   = {[d.name for d in datasets]}")

    client    = OpenAI()
    truncator = Truncator(args.model, args.max_tokens)
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
                client, args.model, "corpus",
                lambda: iter_corpus(corpus_file),
                n, ds_out / "corpus",
                truncator, args.batch_size, args.chunk_batches, logger,
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
                client, args.model, "queries",
                lambda: iter_queries(queries_file),
                n, ds_out / "queries",
                truncator, args.batch_size, args.chunk_batches, logger,
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
    log_path = out_dir / f"embed_beir_log_{ts}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(log_buffer) + "\n")
    print(f"\nrun log written to: {log_path}")


if __name__ == "__main__":
    main()
