"""
Runs evaluation against a LangSmith dataset and records results as an experiment.

Each run creates a new experiment in LangSmith with:
  - Full configuration as metadata (chunking, retrieval, reranking, models)
  - Per-question scores for answer_correctness, hallucination, document_relevance
  - Aggregate scores visible in the LangSmith dashboard

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
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
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


def _score_with_llm(prompt: str) -> float:
    """Ask the LLM to score from 0 to 1. Returns 0.5 on failure."""
    try:
        response = _llm.invoke(prompt)
        text = response.content.strip()
        # Extract first number found in response
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            score = float(numbers[0])
            # Normalize if score is 0-10
            if score > 1:
                score = score / 10
            return min(max(score, 0.0), 1.0)
    except Exception:
        pass
    return 0.5


# ── Evaluators ────────────────────────────────────────────────────────────────

def answer_correctness(run: Run, example: Example) -> dict:
    """Is the answer factually correct compared to the ground truth?"""
    question    = example.inputs.get("question", "")
    answer      = run.outputs.get("answer", "")
    ground_truth = example.outputs.get("ground_truth", "")

    prompt = f"""Rate the factual correctness of the answer compared to the reference.
Question: {question}
Reference: {ground_truth}
Answer: {answer}

Respond with a single number from 0 to 10:
- 10: fully correct and complete
- 5: partially correct
- 0: incorrect or missing key information

Number only:"""

    score = _score_with_llm(prompt)
    return {"key": "answer_correctness", "score": score}


def hallucination(run: Run, example: Example) -> dict:
    """Does the answer contain info NOT in the retrieved context?"""
    question = example.inputs.get("question", "")
    answer   = run.outputs.get("answer", "")
    sources  = " ".join([s.get("content", "") for s in run.outputs.get("sources", [])])

    prompt = f"""Rate whether the answer is grounded in the provided context (no hallucination).
Question: {question}
Context (retrieved documents): {sources[:2000]}
Answer: {answer}

Respond with a single number from 0 to 10:
- 10: answer only uses information from the context (no hallucination)
- 5: answer mostly uses context but adds minor unsupported details
- 0: answer contains significant information not present in context

Number only:"""

    score = _score_with_llm(prompt)
    return {"key": "hallucination", "score": score}


def document_relevance(run: Run, example: Example) -> dict:
    """Are the retrieved documents relevant to the question?"""
    question = example.inputs.get("question", "")
    sources  = " ".join([s.get("content", "") for s in run.outputs.get("sources", [])])

    prompt = f"""Rate how relevant the retrieved documents are to answering the question.
Question: {question}
Retrieved documents: {sources[:2000]}

Respond with a single number from 0 to 10:
- 10: all documents are highly relevant
- 5: some documents are relevant, some are noise
- 0: documents are mostly irrelevant

Number only:"""

    score = _score_with_llm(prompt)
    return {"key": "document_relevance", "score": score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",         required=True)
    parser.add_argument("--dataset-name",    default="cloudrag-eval")
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
            answer_correctness,
            hallucination,
            document_relevance,
        ],
        experiment_prefix=experiment_name,
        metadata=metadata,
        client=client,
        max_concurrency=1,
    )

    print(f"\n{'='*52}")
    print(f"  LANGSMITH RESULTS — {experiment_name}")
    print(f"{'='*52}")
    try:
        import math
        df = results.to_pandas()
        score_cols = [c for c in df.columns if "score" in c.lower()]
        for col in score_cols:
            avg = df[col].dropna().mean()
            if not math.isnan(avg):
                print(f"  {col:<35} {avg:.4f}")
    except Exception:
        print(f"  Results saved — view at LangSmith dashboard")
    print(f"{'='*52}")
    print(f"\n  View at: https://smith.langchain.com\n")


if __name__ == "__main__":
    main()
