"""
End-to-end evaluation using RAGAS 0.1.21.

Unlike the offline MRR evaluation (which measures if the retriever finds
the right chunk), RAGAS measures the full pipeline quality:
  - context_precision   : are the retrieved chunks relevant?
  - context_recall      : were all necessary chunks retrieved?
  - faithfulness        : is the answer grounded in the retrieved context?
  - answer_relevancy    : does the answer actually address the question?

Usage:
    python evaluation/ragas_eval.py --api-url http://localhost:8000
    python evaluation/ragas_eval.py --api-url http://host.docker.internal:8000
    python evaluation/ragas_eval.py --api-url http://<alb_url>

Requirements:
    ragas==0.1.21
    datasets>=2.0.0
    OPENAI_API_KEY must be set in .env
"""

import os
import sys
import json
import argparse
import requests
from typing import List
from dotenv import load_dotenv

# Add project root to path so ingestion/evaluation modules are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ingestion.models import Chunk
from evaluation.dataset import generate_dataset


def _load_chunks(chunks_path: str) -> List[Chunk]:
    with open(chunks_path) as f:
        raw = json.load(f)
    return [Chunk(**c) for c in raw]


def _query_api(api_url: str, question: str, top_k: int = 5) -> dict:
    response = requests.post(
        f"{api_url}/query",
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _build_ragas_dataset(
    api_url: str,
    dataset: List[tuple],
    chunks_by_id: dict,
    max_samples: int = 50,
) -> Dataset:
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    samples = dataset[:max_samples]
    print(f"[ragas] Evaluating {len(samples)} samples...")

    for i, (question, chunk_id) in enumerate(samples):
        print(f"[ragas] Sample {i+1}/{len(samples)}: {question[:60]}...")
        try:
            result = _query_api(api_url, question)
            questions.append(question)
            answers.append(result["answer"])
            contexts.append([s["content"] for s in result["sources"]])
            gt_chunk = chunks_by_id.get(chunk_id)
            ground_truths.append(gt_chunk.content if gt_chunk else "")
        except Exception as e:
            print(f"[ragas] ERROR on sample {i+1}: {e}")
            continue

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",      required=True)
    parser.add_argument("--chunks-path",  default="./output/chunks_structure.json")
    parser.add_argument("--max-samples",  type=int, default=50)
    parser.add_argument("--output",       default="./output/ragas_eval.json")
    args = parser.parse_args()

    print(f"\n{'='*52}")
    print(f"  cloudRAG — RAGAS evaluation")
    print(f"  API     : {args.api_url}")
    print(f"  Chunks  : {args.chunks_path}")
    print(f"  Samples : {args.max_samples}")
    print(f"{'='*52}\n")

    chunks = _load_chunks(args.chunks_path)
    chunks_by_id = {c.chunk_id: c for c in chunks}

    print("[ragas] Generating evaluation dataset...")
    dataset = generate_dataset(chunks[:args.max_samples // 2])

    ragas_dataset = _build_ragas_dataset(
        api_url=args.api_url,
        dataset=dataset,
        chunks_by_id=chunks_by_id,
        max_samples=args.max_samples,
    )

    print("\n[ragas] Running evaluation...")

    # Explicitly configure LLM and embeddings to avoid compatibility issues
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    results = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    os.makedirs("output", exist_ok=True)

    summary = {
        "api_url":           args.api_url,
        "chunks_path":       args.chunks_path,
        "num_samples":       len(ragas_dataset),
        "context_precision": round(float(results["context_precision"]), 4),
        "context_recall":    round(float(results["context_recall"]), 4),
        "faithfulness":      round(float(results["faithfulness"]), 4),
        "answer_relevancy":  round(float(results["answer_relevancy"]), 4),
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  RAGAS RESULTS")
    print(f"{'='*52}")
    print(f"  Context Precision : {summary['context_precision']}")
    print(f"  Context Recall    : {summary['context_recall']}")
    print(f"  Faithfulness      : {summary['faithfulness']}")
    print(f"  Answer Relevancy  : {summary['answer_relevancy']}")
    print(f"{'='*52}")
    print(f"\n  Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
