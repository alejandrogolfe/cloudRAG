"""
Creates a synthetic evaluation dataset in LangSmith.

Run this once to create the dataset. After that you can:
- Add real user queries manually from the LangSmith dashboard
- Add traces from production queries
- Run run_langsmith_eval.py against it at any time

Usage:
    python evaluation/create_langsmith_dataset.py
    python evaluation/create_langsmith_dataset.py --chunks-path ./output/chunks_structure.json --dataset-name "cloudrag-eval-v1"

The dataset is stored in LangSmith and can be reused across experiments.
"""

import os
import sys
import json
import argparse
from typing import List
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langsmith import Client
from ingestion.models import Chunk
from evaluation.dataset import generate_dataset


def _load_chunks(chunks_path: str) -> List[Chunk]:
    with open(chunks_path) as f:
        raw = json.load(f)
    return [Chunk(**c) for c in raw]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-path",
        default="./output/chunks_structure.json",
        help="Path to chunks JSON file used to generate synthetic questions",
    )
    parser.add_argument(
        "--dataset-name",
        default="cloudrag-eval",
        help="Name of the dataset in LangSmith",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=25,
        help="Number of chunks to use for generating questions (2 questions per chunk)",
    )
    args = parser.parse_args()

    print(f"\n{'='*52}")
    print(f"  cloudRAG — create LangSmith dataset")
    print(f"  Dataset : {args.dataset_name}")
    print(f"  Chunks  : {args.chunks_path}")
    print(f"{'='*52}\n")

    # ── 1. Generate synthetic questions ──────────────────────────────────────
    chunks = _load_chunks(args.chunks_path)
    print(f"[dataset] Generating synthetic questions from {args.max_chunks} chunks...")
    dataset = generate_dataset(chunks[:args.max_chunks])
    print(f"[dataset] Generated {len(dataset)} question-chunk pairs")

    # ── 2. Create or update dataset in LangSmith ─────────────────────────────
    client = Client()

    # Check if dataset already exists
    existing = [d for d in client.list_datasets() if d.name == args.dataset_name]

    if existing:
        print(f"[langsmith] Dataset '{args.dataset_name}' already exists — adding new examples")
        dataset_obj = existing[0]
    else:
        dataset_obj = client.create_dataset(
            dataset_name=args.dataset_name,
            description="Synthetic evaluation dataset for cloudRAG — auto-generated from document chunks",
        )
        print(f"[langsmith] Created dataset '{args.dataset_name}'")

    # ── 3. Add examples to dataset ────────────────────────────────────────────
    # Each example has:
    #   inputs:   {"question": "..."}
    #   outputs:  {"ground_truth": "..."} — the chunk content that should answer it
    chunks_by_id = {c.chunk_id: c for c in chunks}
    added = 0

    for question, chunk_id in dataset:
        gt_chunk = chunks_by_id.get(chunk_id)
        if not gt_chunk:
            continue

        client.create_example(
            dataset_id=dataset_obj.id,
            inputs={"question": question},
            outputs={"ground_truth": gt_chunk.content},
            metadata={
                "chunk_id":   chunk_id,
                "chunk_title": gt_chunk.metadata.get("title", ""),
                "source":     gt_chunk.metadata.get("source", ""),
                "strategy":   gt_chunk.metadata.get("strategy", ""),
            },
        )
        added += 1

    print(f"\n[langsmith] Added {added} examples to dataset '{args.dataset_name}'")
    print(f"[langsmith] View at: https://smith.langchain.com → Datasets → {args.dataset_name}\n")


if __name__ == "__main__":
    main()
