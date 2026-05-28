"""
Runs evaluation against a LangSmith dataset and records results as an experiment.

Each run creates a new experiment in LangSmith with:
  - Full configuration as metadata (chunking, retrieval, reranking, models)
  - Per-question scores for answer_correctness, hallucination, document_relevance
  - Aggregate scores visible in the LangSmith dashboard

Evaluators:
  answer_correctness  → LLM compares generated answer vs ground truth
  hallucination       → LLM checks if answer contains info NOT in retrieved context
  document_relevance  → LLM checks if each retrieved document is relevant to the question

Usage:
    python evaluation/run_langsmith_eval.py --api-url http://localhost:8000
    python evaluation/run_langsmith_eval.py --api-url http://<alb_url>
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_openai import ChatOpenAI
from config.chunking import CHUNKING_STRATEGY, EMBEDDING
from config.retrieval import (
    RETRIEVAL_STRATEGY,
    RERANKING_ENABLED,
    RERANKING_MODEL,
    RETRIEVAL_TOP_K_CANDIDATES,
    RETRIEVAL_TOP_K_FINAL,
)

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _query_api(api_url: str, question: str) -> dict:
    response = requests.post(
        f"{api_url}/query",
        json={"question": question, "top_k": RETRIEVAL_TOP_K_FINAL},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _build_experiment_metadata(api_url: str) -> dict:
    return {
        "api_url":            api_url,
        "chunking_strategy":  CHUNKING_STRATEGY,
        "retrieval_strategy": RETRIEVAL_STRATEGY,
        "reranking_enabled":  RERANKING_ENABLED,
        "reranking_model":    RERANKING_MODEL if RERANKING_ENABLED else None,
        "top_k_candidates":   RETRIEVAL_TOP_K_CANDIDATES,
        "top_k_final":        RETRIEVAL_TOP_K_FINAL,
        "embedding_model":    EMBEDDING["model_id"],
        "generator_model":    "gpt-4o",
        "timestamp":          datetime.utcnow().isoformat(),
    }


# ── Evaluators ────────────────────────────────────────────────────────────────

# 1. Answer correctness
# Compares the generated answer against the ground truth chunk.
# Measures factual accuracy — did GPT-4o get it right?
answer_correctness_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "correctness": (
                "Is the answer factually correct and complete based on the reference? "
                "Score 10 if fully correct, 5 if partially correct, 0 if wrong or missing key info."
            )
        },
        "normalize_by": 10,
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs.get("answer", ""),
        "reference":  example.outputs.get("ground_truth", ""),
        "input":      example.inputs.get("question", ""),
    },
)

# 2. Hallucination
# Checks if the answer contains information NOT present in the retrieved context.
# Uses the retrieved sources (not the ground truth) as the reference — this is
# the correct way to detect hallucination in RAG: the model should only use
# what was retrieved, regardless of what the ground truth says.
hallucination_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "faithfulness": (
                "Does the answer contain ONLY information that can be found in the provided context? "
                "Score 10 if the answer is fully grounded in the context (no hallucination). "
                "Score 5 if the answer is mostly grounded but adds minor unsupported details. "
                "Score 0 if the answer contains significant information not present in the context."
            )
        },
        "normalize_by": 10,
    },
    prepare_data=lambda run, example: {
        # Reference is the retrieved context, not the ground truth
        "prediction": run.outputs.get("answer", ""),
        "reference":  " ".join([s.get("content", "") for s in run.outputs.get("sources", [])]),
        "input":      example.inputs.get("question", ""),
    },
)

# 3. Document relevance
# Checks if the retrieved documents are relevant to the question.
# Uses the question (not the ground truth) as reference — this measures
# retrieval quality independently of whether the answer is correct.
document_relevance_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "relevance": (
                "Are the retrieved documents relevant to answer the question? "
                "Score 10 if all documents are highly relevant. "
                "Score 5 if some documents are relevant but there is noise. "
                "Score 0 if the documents are mostly irrelevant to the question."
            )
        },
        "normalize_by": 10,
    },
    prepare_data=lambda run, example: {
        # Prediction is the retrieved documents, reference is the question
        "prediction": " ".join([s.get("content", "") for s in run.outputs.get("sources", [])]),
        "reference":  example.inputs.get("question", ""),
        "input":      example.inputs.get("question", ""),
    },
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",       required=True)
    parser.add_argument("--dataset-name",  default="cloudrag-eval")
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()

    experiment_name = args.experiment_name or (
        f"{CHUNKING_STRATEGY}-{RETRIEVAL_STRATEGY}"
        f"{'-rerank' if RERANKING_ENABLED else ''}"
        f"-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"
    )

    metadata = _build_experiment_metadata(args.api_url)

    print(f"\n{'='*52}")
    print(f"  cloudRAG — LangSmith evaluation")
    print(f"  Dataset    : {args.dataset_name}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Config     : {CHUNKING_STRATEGY} / {RETRIEVAL_STRATEGY} / rerank={RERANKING_ENABLED}")
    print(f"{'='*52}\n")

    def pipeline(inputs: dict) -> dict:
        question = inputs["question"]
        print(f"  → {question[:70]}...")
        result = _query_api(args.api_url, question)
        return {
            "answer":  result["answer"],
            "sources": result["sources"],
        }

    client = Client()

    results = evaluate(
        pipeline,
        data=args.dataset_name,
        evaluators=[
            answer_correctness_evaluator,
            hallucination_evaluator,
            document_relevance_evaluator,
        ],
        experiment_prefix=experiment_name,
        metadata=metadata,
        client=client,
    )

    print(f"\n{'='*52}")
    print(f"  LANGSMITH RESULTS — {experiment_name}")
    print(f"{'='*52}")
    for metric, score in results.aggregate_feedback.items():
        print(f"  {metric:<30} {score:.4f}")
    print(f"{'='*52}")
    print(f"\n  View at: https://smith.langchain.com\n")


if __name__ == "__main__":
    main()
