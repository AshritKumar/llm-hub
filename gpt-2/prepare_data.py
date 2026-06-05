"""
Download, tokenize, and shard datasets for GPT-2 training.

Datasets:
  1. FineWeb-Edu 10BT sample  (55% of training mix)
  2. PleIAs/LoC-PD-Books       (25% — novels)
  3. TinyStories               (20%)

Each dataset is tokenized with the GPT-2 BPE tokenizer (tiktoken) and
saved as uint16 numpy shards (~100M tokens each, ~200 MB).

Why shard?
  Large datasets don't fit in RAM (FineWeb-Edu 10BT ≈ 20 GB tokenized).
  Sharding lets us load one chunk at a time during training.

This script is ONLY for data preparation (download + tokenize + shard).
The mixing weights (55/25/20) are stored here as metadata but are only
applied at training time by ShardedDataLoader in gpt2_model.py.

Usage:
  python prepare_data.py                      # prepare all datasets
  python prepare_data.py --dataset fineweb    # prepare only FineWeb-Edu
  python prepare_data.py --dataset books      # prepare only books
  python prepare_data.py --dataset tinystories
  python prepare_data.py --max_tokens 500_000_000  # cap tokens per dataset

Deps:
  pip install datasets tiktoken numpy
"""

import os
import argparse
import glob
import numpy as np
import tiktoken

# HuggingFace `datasets` library — provides load_dataset() which can stream
# data directly from HuggingFace Hub over HTTP without downloading full files.
# Think of it like a smart iterator over the dataset rows.
from datasets import load_dataset

# GPT-2 vocab is 50257 tokens — fits in uint16 (max 65535).
# uint16 halves disk usage vs int32 (~200 MB per 100M-token shard).
SHARD_SIZE = int(100e6)  # 100M tokens per shard

# All shard files go under  gpt2_replica/data/{dataset_name}/shard_NNNNNN.npy
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry maps a short name to its HuggingFace coordinates.
#   hf_path   : HF dataset id  (passed to load_dataset as the first arg)
#               e.g. load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
#   hf_config : Some datasets have multiple subsets (configs).
#               FineWeb-Edu has "sample-10BT", "sample-100BT", "default", etc.
#               None means the dataset has only one config.
#   text_field: The column name that contains the actual text.
#               Each HF dataset has different column names — we need to know which one holds the text.
#   split     : "train", "validation", "test" — which split to stream.
#   weight    : Target fraction of training batches (NOT used by this script).
#               Only used later by ShardedDataLoader at training time.
#               Over many training steps: ~55% of batches will come from FineWeb,
#               ~25% from books, ~20% from TinyStories.
#
# Training mix:  FineWeb-Edu 55%  |  Novels 25%  |  TinyStories 20%
DATASET_REGISTRY = {
    "fineweb": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_config": "sample-10BT",        # 10 billion token sample (~28 GB compressed on HF)
        "text_field": "text",
        "split": "train",
        "weight": 0.55,
    },
    "books": {
        "hf_path": "storytracer/LoC-PD-Books",  # ~140k US public-domain novels from Library of Congress
        "hf_config": None,
        "text_field": "text",
        "split": "train",
        "weight": 0.25,
    },
    "tinystories": {
        "hf_path": "roneneldan/TinyStories",    # ~2.1M short children's stories (GPT-3.5/4 generated)
        "hf_config": None,
        "text_field": "text",
        "split": "train",
        "weight": 0.20,
    },
}


def tokenize_and_shard(dataset_name: str, max_tokens: int | None = None):
    """Stream a HF dataset, tokenize each document, and flush full shards to disk.

    Pipeline per document:
      1. Pull text from the streaming iterator (no full download needed)
      2. Tokenize with tiktoken's encode_ordinary (skips special tokens)
      3. Append an <|endoftext|> (EOT) token as a document separator
      4. Fill a numpy buffer; once the buffer hits SHARD_SIZE, flush to .npy

    Args:
        dataset_name: key into DATASET_REGISTRY
        max_tokens:   Stop after this many tokens (None = process entire dataset).
                      FineWeb-Edu 10BT would produce ~100 shards (~20 GB on disk).
                      Use --max_tokens to limit, e.g. 500M tokens = 5 shards ≈ 1 GB.
                      Good for test runs or when disk space is limited.
    """
    cfg = DATASET_REGISTRY[dataset_name]
    out_dir = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    # EOT (<|endoftext|>, token 50256) is inserted between every document.
    # This teaches the model that context resets here — without it, the model would
    # treat the last sentence of document A and first sentence of document B as continuous text.
    eot = enc.eot_token

    print(f"\n{'='*60}")
    print(f"Preparing '{dataset_name}'  ({cfg['hf_path']})")
    print(f"  shard size : {SHARD_SIZE:,} tokens")
    print(f"  max tokens : {'unlimited' if max_tokens is None else f'{max_tokens:,}'}")
    print(f"  output dir : {out_dir}")
    print(f"{'='*60}")

    # load_dataset() with streaming=True returns an IterableDataset.
    # Instead of downloading the full dataset (FineWeb = ~28 GB on HF), it fetches
    # data in small HTTP chunks on demand. Each `for example in ds:` pulls the next row.
    # This means RAM usage stays constant regardless of dataset size.
    ds = load_dataset(
        cfg["hf_path"],        # e.g. "HuggingFaceFW/fineweb-edu"
        name=cfg["hf_config"], # e.g. "sample-10BT" (subset), or None
        split=cfg["split"],    # e.g. "train"
        streaming=True,        # stream over HTTP, don't download to disk
    )

    shard_idx = 0
    # Pre-allocate a fixed-size buffer so we're not constantly resizing a Python list
    token_buf = np.empty(SHARD_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    doc_count = 0

    for example in ds:
        text = example[cfg["text_field"]]
        if text is None or len(text) == 0:
            continue

        # encode_ordinary vs encode:
        #   encode("Hello <|endoftext|> world") → would parse <|endoftext|> as special token 50256
        #   encode_ordinary("Hello <|endoftext|> world") → treats it as literal text, no special parsing
        # We use encode_ordinary so raw text containing "<|endoftext|>" isn't misinterpreted,
        # then we explicitly append the real EOT token ourselves as a document separator.
        tokens = enc.encode_ordinary(text)
        tokens.append(eot)
        n = len(tokens)

        # If we'd exceed max_tokens, truncate this document to fit exactly
        if max_tokens is not None and total_tokens + n > max_tokens:
            n = max_tokens - total_tokens
            tokens = tokens[:n]
            if n == 0:
                break

        # Fill the shard buffer, flushing whenever it's full.
        # A single document's tokens may span across shard boundaries, e.g.:
        #   shard_003 = [..., doc_A_part1]   ← buffer fills up mid-document
        #   shard_004 = [doc_A_part2, ...]   ← rest of doc_A continues here
        # This is fine — the model sees a flat token stream anyway.
        remaining = tokens
        while len(remaining) > 0:
            space = SHARD_SIZE - buf_pos        # how many tokens fit before this shard is full
            chunk = remaining[:space]            # take as many as will fit
            token_buf[buf_pos : buf_pos + len(chunk)] = chunk
            buf_pos += len(chunk)
            remaining = remaining[space:]        # leftover goes into the next shard

            if buf_pos == SHARD_SIZE:            # shard is full — flush to disk
                shard_path = os.path.join(out_dir, f"shard_{shard_idx:06d}.npy")
                np.save(shard_path, token_buf)   # saves as data/{dataset_name}/shard_000000.npy
                print(f"  shard {shard_idx:>4d}  |  {SHARD_SIZE:>12,} tokens  |  total {total_tokens + buf_pos:>14,}")
                shard_idx += 1
                buf_pos = 0

        total_tokens += n
        doc_count += 1

        if doc_count % 10_000 == 0:
            print(f"  ... {doc_count:>8,} docs  |  {total_tokens:>14,} tokens")

        if max_tokens is not None and total_tokens >= max_tokens:
            break

    # Flush whatever remains in the buffer as a final (smaller) shard
    if buf_pos > 0:
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:06d}.npy")
        np.save(shard_path, token_buf[:buf_pos])
        print(f"  shard {shard_idx:>4d}  |  {buf_pos:>12,} tokens  (final)")
        shard_idx += 1

    print(f"\nDone: '{dataset_name}' — {doc_count:,} docs, {total_tokens:,} tokens, {shard_idx} shards")
    return total_tokens


def assign_splits(dataset_name: str):
    """Rename shard files to train_shard_*, val_shard_*, test_shard_*.

    The ShardedDataLoader uses the filename prefix to load the correct split.

    Strategy:
      - 3+ shards : last → test, second-to-last → val, rest → train
      - 2 shards  : first → train, second split in half → val + test
      - 1 shard   : split the data 90 / 5 / 5 into three files
    """
    out_dir = os.path.join(DATA_DIR, dataset_name)
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.npy")))
    n = len(shards)

    if n == 0:
        return

    if n >= 3:
        for i, path in enumerate(shards[:-2]):
            os.rename(path, os.path.join(out_dir, f"train_shard_{i:06d}.npy"))
        os.rename(shards[-2], os.path.join(out_dir, "val_shard_000000.npy"))
        os.rename(shards[-1], os.path.join(out_dir, "test_shard_000000.npy"))
    elif n == 2:
        os.rename(shards[0], os.path.join(out_dir, "train_shard_000000.npy"))
        data = np.load(shards[1])
        mid = len(data) // 2
        np.save(os.path.join(out_dir, "val_shard_000000.npy"), data[:mid])
        np.save(os.path.join(out_dir, "test_shard_000000.npy"), data[mid:])
        os.remove(shards[1])
    else:  # n == 1
        data = np.load(shards[0])
        total = len(data)
        train_end = int(total * 0.9)
        val_end   = int(total * 0.95)
        np.save(os.path.join(out_dir, "train_shard_000000.npy"), data[:train_end])
        np.save(os.path.join(out_dir, "val_shard_000000.npy"), data[train_end:val_end])
        np.save(os.path.join(out_dir, "test_shard_000000.npy"), data[val_end:])
        os.remove(shards[0])

    # Print what we created
    for split in ("train", "val", "test"):
        split_shards = sorted(glob.glob(os.path.join(out_dir, f"{split}_shard_*.npy")))
        tok = sum(len(np.load(s)) for s in split_shards)
        print(f"  {split:<5s}: {len(split_shards)} shard(s), {tok:,} tokens")


def list_shards(dataset_name: str, split: str = None) -> list[str]:
    shard_dir = os.path.join(DATA_DIR, dataset_name)
    if split:
        return sorted(glob.glob(os.path.join(shard_dir, f"{split}_shard_*.npy")))
    # Return all split shards
    return sorted(glob.glob(os.path.join(shard_dir, "*_shard_*.npy")))


def print_summary():
    print(f"\n{'='*60}")
    print("Dataset summary")
    print(f"{'='*60}")
    for name in DATASET_REGISTRY:
        all_shards = list_shards(name)
        if not all_shards:
            print(f"  {name:<15s}  —  not prepared yet")
            continue
        for split in ("train", "val", "test"):
            split_shards = list_shards(name, split)
            if split_shards:
                tok = sum(len(np.load(s)) for s in split_shards)
                print(f"  {name:<15s}  {split:<5s}: {len(split_shards):>4d} shard(s)  |  {tok:>14,} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for GPT-2")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASET_REGISTRY.keys()),
                        help="Prepare only this dataset (default: all)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Cap tokens per dataset (useful for quick test runs)")
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASET_REGISTRY.keys())

    for ds_name in targets:
        tokenize_and_shard(ds_name, max_tokens=args.max_tokens)
        assign_splits(ds_name)

    print_summary()


#   ── Run commands ────────────────────────────────────────────────────────────
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  QUICK TEST — small amount of data to verify the pipeline works       │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   # 10M tokens per dataset (~1 shard each, ~200 MB total, takes ~1-2 min):
#   python prepare_data.py --max_tokens 10_000_000
#
#   # Test just one dataset:
#   python prepare_data.py --dataset tinystories --max_tokens 10_000_000
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  SINGLE DATASET — download and shard one dataset at a time            │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   python prepare_data.py --dataset fineweb            # FineWeb-Edu 10BT (~10B tokens, ~20 GB, takes hours)
#   python prepare_data.py --dataset books              # LoC-PD-Books (~140k novels)
#   python prepare_data.py --dataset tinystories        # TinyStories (~2.1M stories)
#
#   # Cap each to 500M tokens (~1 GB on disk):
#   python prepare_data.py --dataset fineweb --max_tokens 500_000_000
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  ALL DATASETS — prepare everything for a full training run            │
#   └─────────────────────────────────────────────────────────────────────────┘
#
#   python prepare_data.py                               # full size (can take many hours)
#   python prepare_data.py --max_tokens 500_000_000      # cap each at 500M tokens
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │  OUTPUT STRUCTURE                                                     │
#   │                                                                       │
#   │  data/                                                                │
#   │  ├── fineweb/                                                         │
#   │  │   ├── shard_000000.npy    (~200 MB, 100M uint16 tokens)           │
#   │  │   ├── shard_000001.npy                                            │
#   │  │   └── ...                                                         │
#   │  ├── books/                                                           │
#   │  │   ├── shard_000000.npy                                            │
#   │  │   └── ...                                                         │
#   │  └── tinystories/                                                     │
#   │      ├── shard_000000.npy                                            │
#   │      └── ...                                                         │
#   │                                                                       │
#   │  After shards are ready, train with:                                  │
#   │  python train_gpt2.py --sharded                                      │
#   └─────────────────────────────────────────────────────────────────────────┘
