"""
Local pipeline: chunk → embed → evaluate (no OpenSearch needed).

Usage:
    python run_local.py --docs-path ./data/

This script:
1. Runs the ingestion pipeline (load, filter, clean, chunk, embed)
2. Saves chunks+embeddings to ./output/chunks_{strategy}.json
3. Generates a synthetic evaluation dataset
4. Computes MRR and Hit Rate (retrieval quality)
5. Computes Faithfulness (chunk completeness quality)
6. Prints a full report

Interpreting results:
  MRR high + Faithfulness high   → chunking strategy is good
  MRR high + Faithfulness low    → retriever works but chunks are cut/incomplete
  MRR low  + Faithfulness high   → chunks are good but retriever struggles to find them
  MRR low  + Faithfulness low    → both chunking and retrieval need work

When you're happy with results, upload to OpenSearch:
    python run_upload.py --chunks-path ./output/chunks_{strategy}.json
"""

import os
import sys
import json
import argparse
import dataclasses
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from ingestion.graph import build_ingestion_graph
from evaluation.dataset import generate_dataset
from evaluation.retriever import retrieve
from evaluation.metrics import evaluate
from evaluation.faithfulness import evaluate_faithfulness
from config.chunking import CHUNKING_STRATEGY, EMBEDDING
openai_client = OpenAI()


def _embed_query(text: str):
    response = openai_client.embeddings.create(
        model=EMBEDDING["model_id"],
        input=text,
    )
    return response.data[0].embedding


def _save_chunks(chunks, strategy: str) -> str:
    os.makedirs("output", exist_ok=True)
    path = f"output/chunks_{strategy}.json"
    with open(path, "w") as f:
        json.dump([dataclasses.asdict(c) for c in chunks], f, indent=2)
    print(f"[save] Chunks saved to {path}")
    return path


def _save_report(report: dict, strategy: str) -> str:
    os.makedirs("output", exist_ok=True)
    path = f"output/eval_{strategy}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[save] Report saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs-path",
        default="/opt/project/data/",
        help="Path to the folder containing markdown files",
    )
    args = parser.parse_args()

    strategy = CHUNKING_STRATEGY
    print(f"\n{'='*52}")
    print(f"  cloudRAG — local pipeline")
    print(f"  Strategy : {strategy}")
    print(f"  Docs     : {args.docs_path}")
    print(f"{'='*52}\n")

    # ── 1. Ingestion ───────────────────────────────────────
    graph = build_ingestion_graph()
    state = graph.invoke({
        "docs_path": args.docs_path,
        "strategy": strategy,
        "raw_documents": [],
        "filtered_documents": [],
        "filtered_out": [],
        "cleaned_documents": [],
        "chunks": [],
        "embedded_chunks": [],
    })

    chunks = state["embedded_chunks"]
    if not chunks:
        print("No chunks produced. Check your docs path and filter settings.")
        sys.exit(1)

    # ── 2. Save chunks ─────────────────────────────────────
    _save_chunks(chunks, strategy)

    # ── 3. Generate synthetic dataset ─────────────────────
    print("\n[eval] Generating synthetic queries...")
    dataset = generate_dataset(chunks)  # [(query, chunk_id), ...]

    # ── 4. Embed queries ───────────────────────────────────
    print("\n[eval] Embedding queries...")
    query_embeddings = [_embed_query(q) for q, _ in dataset]

    # ── 5. MRR + Hit Rate ──────────────────────────────────
    print("\n[eval] Computing MRR and Hit Rate...")
    retrieval_report = evaluate(dataset, query_embeddings, chunks)

    # ── 6. Faithfulness ────────────────────────────────────
    # Get top-1 retrieved chunk per query for faithfulness eval
    print("\n[eval] Computing Faithfulness...")
    top1_chunks = []
    for query_embedding in query_embeddings:
        results = retrieve(query_embedding, chunks, top_k=1)
        top1_chunks.append(results[0][0])

    faithfulness_report = evaluate_faithfulness(dataset, top1_chunks)

    # ── 7. Save and print full report ─────────────────────
    full_report = {
        "strategy": strategy,
        "retrieval": retrieval_report,
        "faithfulness": faithfulness_report,
    }
    _save_report(full_report, strategy)

    print(f"\n{'='*52}")
    print(f"  RESULTS — strategy: {strategy}")
    print(f"{'='*52}")
    print(f"  — Retrieval —")
    print(f"  MRR:                  {retrieval_report['mrr']}")
    print(f"  Hit Rate @ {retrieval_report['top_k']}:        {retrieval_report['hit_rate']}")
    print(f"  — Faithfulness —")
    print(f"  Avg Faithfulness:     {faithfulness_report['avg_faithfulness']}")
    print(f"  Fully answered:       {faithfulness_report['fully_answered']}/{faithfulness_report['total_queries']}")
    print(f"  Partially answered:   {faithfulness_report['partially_answered']}/{faithfulness_report['total_queries']}  <- possible cut chunks")
    print(f"  Not answered:         {faithfulness_report['not_answered']}/{faithfulness_report['total_queries']}")
    print(f"{'='*52}\n")

    # Show possibly cut chunks
    cut_chunks = [d for d in faithfulness_report["details"] if d["possibly_cut"]]
    if cut_chunks:
        print(f"  Possibly cut chunks ({len(cut_chunks)}):")
        for d in cut_chunks[:5]:
            print(f"    Query:  {d['query']}")
            print(f"    Reason: {d['reason']}\n")

    # Show missed retrieval queries
    missed = [d for d in retrieval_report["details"] if not d["hit"]]
    if missed:
        print(f"  Missed retrieval queries ({len(missed)}):")
        for d in missed[:5]:
            print(f"    - {d['query']}")
    print()


if __name__ == "__main__":
    main()
