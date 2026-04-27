"""
Embed an entire BEIR-style dataset via OpenRouter's embeddings API.
OpenRouter exposes an OpenAI-compatible /v1/embeddings endpoint.

Supported embedding models on OpenRouter (as of Apr 2026):
  sentence-transformers/all-minilm-l12-v2   384-dim   $0.005/M tokens
  qwen/qwen3-embedding-8b                   4096-dim  $0.010/M tokens
  qwen/qwen3-embedding-4b                   2560-dim  $0.010/M tokens

Auth:
  Set OPENROUTER_API_KEY in .env.
  Get one at https://openrouter.ai/settings/keys

Same chunk layout / resumability / log conventions as embed_beir.py.
Chunk filenames carry a model tag (e.g. "minilm-l12" or "qwen3-emb-8b")
so outputs from different models never collide.

Usage:
  # MiniLM-L12 (cheapest, 384-dim, comparable to MiniLM-L6)
  python embed_beir_openrouter.py \\
      --data-dir ~/bier-data --datasets fever \\
      --model sentence-transformers/all-minilm-l12-v2 \\
      --output-dir ./embeddings_minilm_or

  # Qwen3-Embedding-8B (best quality, 4096-dim)
  python embed_beir_openrouter.py \\
      --data-dir ~/bier-data --datasets fever \\
      --model qwen/qwen3-embedding-8b \\
      --output-dir ./embeddings_qwen3_8b

  # Qwen3-Embedding-4B (mid quality, 2560-dim)
  python embed_beir_openrouter.py \\
      --data-dir ~/bier-data --datasets fever \\
      --model qwen/qwen3-embedding-4b \\
      --output-dir ./embeddings_qwen3_4b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI           # OpenRouter is OpenAI-compatible
from tqdm import tqdm

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Conservative batch size -- OpenRouter may have payload size limits
DEFAULT_BATCH   = 32
DEFAULT_WORKERS = 8       # concurrent API calls in flight
CHUNK_BATCHES   = 40
MAX_RETRIES     = 8
INITIAL_BACKOFF = 2.0
MAX_BACKOFF     = 120.0

# Model → (default tag, expected dim)
MODEL_DEFAULTS = {
    "sentence-transformers/all-minilm-l12-v2": ("minilm-l12-or",  384),
    "qwen/qwen3-embedding-8b":                 ("qwen3-emb-8b",  4096),
    "qwen/qwen3-embedding-4b":                 ("qwen3-emb-4b",  2560),
}


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

    logger = logging.getLogger("embed_beir_openrouter")
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
    title = (rec.get("title") or "").strip()
    text  = (rec.get("text")  or "").strip()
    if title and text:
        combined = f"{title}\n{text}"
    else:
        combined = title or text
    return combined if combined else "[empty]"


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
            text = (rec.get("text") or "").strip()
            yield Item(str(rec["_id"]), text if text else "[empty]")


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Chunk layout + resume  (identical to other embed_beir scripts)
# ---------------------------------------------------------------------------

def chunk_path(out_dir: Path, idx: int, tag: str) -> Path:
    return out_dir / f"chunk_{idx:07d}.{tag}.npy"


def chunk_ids_path(out_dir: Path, idx: int, tag: str) -> Path:
    return out_dir / f"chunk_{idx:07d}.{tag}.ids.txt"


def scan_existing(out_dir: Path, tag: str,
                  logger: logging.Logger) -> tuple[int, int]:
    if not out_dir.exists():
        return 0, 0
    chunks = sorted(out_dir.glob(f"chunk_*.{tag}.npy"))
    processed = good = 0
    for p in chunks:
        stem  = p.name[:-len(".npy")]
        ids_p = out_dir / (stem + ".ids.txt")
        if not ids_p.exists():
            logger.warning(f"dropping orphan chunk {p.name}")
            p.unlink()
            continue
        try:
            arr   = np.load(p, mmap_mode="r")
            n_ids = sum(1 for _ in ids_p.open("r", encoding="utf-8"))
        except Exception as e:
            logger.warning(f"dropping unreadable chunk {p.name}: {e}")
            p.unlink(); ids_p.unlink(missing_ok=True)
            continue
        if arr.shape[0] != n_ids:
            logger.warning(f"dropping mismatched chunk {p.name}")
            p.unlink(); ids_p.unlink(missing_ok=True)
            continue
        processed += int(arr.shape[0])
        good += 1
    return good, processed


def flush_chunk(out_dir: Path, idx: int, tag: str,
                vecs: list[np.ndarray], ids: list[str]) -> None:
    if not vecs:
        return
    arr   = np.concatenate(vecs, axis=0).astype(np.float32, copy=False)
    npy_p = chunk_path(out_dir, idx, tag)
    ids_p = chunk_ids_path(out_dir, idx, tag)
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
# Embedding via OpenRouter
# ---------------------------------------------------------------------------

def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE,
        default_headers={
            "HTTP-Referer": "https://github.com/IRC-research",
            "X-Title": "BEIR-IRC-embedding",
        },
    )


def embed_batch(client: OpenAI, model: str, texts: list[str],
                logger: logging.Logger) -> np.ndarray:
    delay    = INITIAL_BACKOFF
    last_err = None

    # If batch has multiple texts and keeps failing, try splitting it in half
    # down to individual items before giving up.
    if len(texts) > 1:
        for attempt in range(4):
            try:
                resp = client.embeddings.create(model=model, input=texts,
                                                encoding_format="float")
                if resp.data is None or len(resp.data) == 0:
                    logger.warning(f"null data response (batch={len(texts)}): "
                                   f"{vars(resp) if hasattr(resp, '__dict__') else resp}")
                    raise ValueError("empty/null data")
                vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / np.clip(norms, 1e-12, None)
            except Exception as e:
                last_err = e
                logger.warning(f"batch embed error (attempt {attempt+1}/4, "
                               f"batch={len(texts)}): {type(e).__name__}: {e}. "
                               f"sleep {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        # Full batch failed -- fall back to one-by-one
        logger.warning(f"batch of {len(texts)} failed 4x -- falling back to one-by-one")
        vecs_list = [embed_batch(client, model, [t], logger) for t in texts]
        return np.concatenate(vecs_list, axis=0)

    # Single-item path -- retry up to MAX_RETRIES
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.embeddings.create(model=model, input=texts,
                                            encoding_format="float")
            if resp.data is None or len(resp.data) == 0:
                logger.warning(f"null data response (single): "
                               f"{vars(resp) if hasattr(resp, '__dict__') else resp}")
                raise ValueError("empty/null data")
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.clip(norms, 1e-12, None)
        except Exception as e:
            last_err = e
            logger.warning(f"single embed error (attempt {attempt+1}/{MAX_RETRIES}): "
                           f"{type(e).__name__}: {e}. sleep {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, MAX_BACKOFF)
    raise RuntimeError(f"embed failed after {MAX_RETRIES} retries: {last_err}")


def embed_split(
    client: OpenAI,
    model: str,
    tag: str,
    split_name: str,
    items_iter_factory,
    total_items: int,
    out_dir: Path,
    batch_size: int,
    chunk_batches: int,
    logger: logging.Logger,
    n_workers: int = DEFAULT_WORKERS,
) -> dict:
    """
    Concurrent embedding: a producer feeds batches into a queue;
    n_workers threads each call the API in parallel; a consumer thread
    collects results in order and flushes chunks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "DONE"
    stats = {"split": split_name, "total": total_items,
             "skipped": 0, "embedded": 0, "batches": 0, "seconds": 0.0}

    if done_marker.exists():
        logger.info(f"  {split_name}: already complete. skipping.")
        stats["skipped"] = total_items
        return stats

    n_chunks_done, already = scan_existing(out_dir, tag, logger)
    stats["skipped"] = already
    if already:
        logger.info(f"  {split_name}: resuming after {already} items "
                    f"in {n_chunks_done} chunks")

    chunk_size = batch_size * chunk_batches

    # Each work item is (seq_no, ids, texts).
    # Results queue holds (seq_no, ids, vecs) so we can reorder.
    work_q:   queue.Queue = queue.Queue(maxsize=n_workers * 4)
    result_q: queue.Queue = queue.Queue()
    error_box: list[Exception] = []

    def worker():
        while True:
            item = work_q.get()
            if item is None:
                work_q.task_done()
                break
            seq, ids, texts = item
            try:
                vecs = embed_batch(client, model, texts, logger)
                result_q.put((seq, ids, vecs))
            except Exception as e:
                error_box.append(e)
                result_q.put((seq, ids, None))   # signal failure
            finally:
                work_q.task_done()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(n_workers)]
    for t in threads:
        t.start()

    t0   = time.perf_counter()
    pbar = tqdm(total=total_items, initial=already, desc=f"{split_name:<7}",
                unit="doc", dynamic_ncols=True)

    # Consumer: collect results in order, flush chunks
    chunk_idx  = n_chunks_done
    buf_vecs:  list[np.ndarray] = []
    buf_ids:   list[str]        = []
    pending:   dict[int, tuple] = {}   # out-of-order buffer
    next_seq   = 0
    total_sent = 0    # batches enqueued
    total_recv = 0    # batches collected

    def consume_ready():
        nonlocal chunk_idx, buf_vecs, buf_ids, total_recv
        while next_seq in pending:
            ids, vecs = pending.pop(next_seq)
            if vecs is None:
                raise RuntimeError(f"worker failed on batch seq={next_seq}")
            buf_vecs.append(vecs)
            buf_ids.extend(ids)
            stats["batches"]  += 1
            stats["embedded"] += len(ids)
            pbar.update(len(ids))
            total_recv += 1
            if sum(v.shape[0] for v in buf_vecs) >= chunk_size:
                flush_chunk(out_dir, chunk_idx, tag, buf_vecs, buf_ids)
                logger.debug(f"  flushed chunk {chunk_idx}")
                chunk_idx += 1
                buf_vecs, buf_ids = [], []

    # Producer: read items, build batches, enqueue
    batch_texts: list[str] = []
    batch_ids:   list[str] = []
    seen = 0

    try:
        for item in items_iter_factory():
            seen += 1
            if seen <= already:
                continue
            batch_ids.append(item.ident)
            batch_texts.append(item.text)   # always non-empty; guaranteed by iter_*
            if len(batch_texts) >= batch_size:
                work_q.put((total_sent, batch_ids, batch_texts))
                total_sent += 1
                batch_texts, batch_ids = [], []
                # drain results that have arrived
                while not result_q.empty():
                    seq, ids, vecs = result_q.get_nowait()
                    pending[seq] = (ids, vecs)
                next_seq_before = next_seq
                while next_seq in pending:
                    ids, vecs = pending.pop(next_seq)
                    if vecs is None:
                        raise RuntimeError("worker embed failed")
                    buf_vecs.append(vecs)
                    buf_ids.extend(ids)
                    stats["batches"]  += 1
                    stats["embedded"] += len(ids)
                    pbar.update(len(ids))
                    total_recv += 1
                    if sum(v.shape[0] for v in buf_vecs) >= chunk_size:
                        flush_chunk(out_dir, chunk_idx, tag, buf_vecs, buf_ids)
                        logger.debug(f"  flushed chunk {chunk_idx}")
                        chunk_idx += 1
                        buf_vecs, buf_ids = [], []
                    next_seq += 1  # noqa: SIM113

        # flush remaining partial batch
        if batch_texts:
            work_q.put((total_sent, batch_ids, batch_texts))
            total_sent += 1

        # signal workers to stop
        for _ in threads:
            work_q.put(None)
        for t in threads:
            t.join()

        # drain result queue
        while total_recv < total_sent:
            seq, ids, vecs = result_q.get()
            pending[seq] = (ids, vecs)
            while next_seq in pending:
                ids2, vecs2 = pending.pop(next_seq)
                if vecs2 is None:
                    raise RuntimeError("worker embed failed")
                buf_vecs.append(vecs2)
                buf_ids.extend(ids2)
                stats["batches"]  += 1
                stats["embedded"] += len(ids2)
                pbar.update(len(ids2))
                total_recv += 1
                if sum(v.shape[0] for v in buf_vecs) >= chunk_size:
                    flush_chunk(out_dir, chunk_idx, tag, buf_vecs, buf_ids)
                    logger.debug(f"  flushed chunk {chunk_idx}")
                    chunk_idx += 1
                    buf_vecs, buf_ids = [], []
                next_seq += 1

        if buf_vecs:
            flush_chunk(out_dir, chunk_idx, tag, buf_vecs, buf_ids)
        if error_box:
            raise error_box[0]
        done_marker.write_text("ok\n")

    finally:
        pbar.close()

    stats["seconds"] = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_datasets(data_dir: Path,
                      filter_names: list[str] | None) -> list[Path]:
    out = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if filter_names and child.name not in filter_names:
            continue
        if (child / "corpus.jsonl").exists() or (child / "queries.jsonl").exists():
            out.append(child)
        else:
            out.extend(g for g in child.iterdir()
                       if g.is_dir() and (g / "corpus.jsonl").exists())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default=os.environ.get("BEIR_DATA_DIR", "bier-data"))
    ap.add_argument("--output-dir", default=None,
                    help="Output directory. Defaults to ./embeddings_<model-tag>.")
    ap.add_argument("--datasets",   nargs="*", default=None)
    ap.add_argument("--model",
                    default="sentence-transformers/all-minilm-l12-v2",
                    help="OpenRouter model ID. Common choices:\n"
                         "  sentence-transformers/all-minilm-l12-v2\n"
                         "  qwen/qwen3-embedding-8b\n"
                         "  qwen/qwen3-embedding-4b")
    ap.add_argument("--tag",        default=None,
                    help="Chunk filename tag. Auto-derived from model if omitted.")
    ap.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH)
    ap.add_argument("--workers",       type=int, default=DEFAULT_WORKERS,
                    help=f"Concurrent API calls in flight (default {DEFAULT_WORKERS}).")
    ap.add_argument("--chunk-batches", type=int, default=CHUNK_BATCHES)
    ap.add_argument("--skip-corpus",   action="store_true")
    ap.add_argument("--skip-queries",  action="store_true")
    ap.add_argument("--quiet",         action="store_true")
    args = ap.parse_args()

    logger, log_buffer = build_logger(verbose=not args.quiet)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    # Derive tag from model name if not provided
    tag = args.tag
    if tag is None:
        tag = MODEL_DEFAULTS.get(args.model, (args.model.split("/")[-1][:20], None))[0]

    out_dir_default = f"embeddings_{tag}"
    out_dir  = Path(args.output_dir or out_dir_default).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        logger.error(f"data dir not found: {data_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(data_dir, args.datasets)
    if not datasets:
        logger.error(f"no BEIR datasets found under {data_dir}")
        sys.exit(1)

    client = make_client(api_key)

    expected_dim = MODEL_DEFAULTS.get(args.model, (None, "?"))[1]
    logger.info(f"model      = {args.model}  (tag={tag}  expected_dim={expected_dim})")
    logger.info(f"data_dir   = {data_dir}")
    logger.info(f"output_dir = {out_dir}")
    logger.info(f"batch_size = {args.batch_size}  workers = {args.workers}  chunk_batches = {args.chunk_batches}")
    logger.info(f"datasets   = {[d.name for d in datasets]}")

    run_start  = time.perf_counter()
    all_stats: list[dict] = []

    for ds_path in datasets:
        ds_name = ds_path.name
        ds_out  = out_dir / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n=== dataset: {ds_name} ===")

        corpus_file  = ds_path / "corpus.jsonl"
        queries_file = ds_path / "queries.jsonl"

        if not args.skip_corpus and corpus_file.exists():
            n = count_lines(corpus_file)
            logger.info(f"  corpus: {n:,} docs")
            s = embed_split(client, args.model, tag, "corpus",
                            lambda: iter_corpus(corpus_file),
                            n, ds_out / "corpus",
                            args.batch_size, args.chunk_batches, logger,
                            n_workers=args.workers)
            s["dataset"] = ds_name
            all_stats.append(s)

        if not args.skip_queries and queries_file.exists():
            n = count_lines(queries_file)
            logger.info(f"  queries: {n:,}")
            s = embed_split(client, args.model, tag, "queries",
                            lambda: iter_queries(queries_file),
                            n, ds_out / "queries",
                            args.batch_size, args.chunk_batches, logger,
                            n_workers=args.workers)
            s["dataset"] = ds_name
            all_stats.append(s)

    total_secs = time.perf_counter() - run_start

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'dataset':<25} {'split':<8} {'total':>10} {'skipped':>10} "
                f"{'embedded':>10} {'batches':>8} {'sec':>8}")
    for s in all_stats:
        logger.info(f"{s['dataset']:<25} {s['split']:<8} {s['total']:>10} "
                    f"{s['skipped']:>10} {s['embedded']:>10} "
                    f"{s['batches']:>8} {s['seconds']:>8.1f}")
    logger.info(f"total wall clock: {total_secs:.1f} s")

    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"embed_beir_openrouter_log_{ts}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(log_buffer) + "\n")
    print(f"\nrun log -> {log_path}")


if __name__ == "__main__":
    main()
