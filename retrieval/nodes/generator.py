"""
Generator node: calls GPT-4o with the assembled prompt and returns the answer.
"""

from openai import OpenAI
from retrieval.state import RetrievalState
from langsmith import traceable

client = OpenAI()


@traceable(name="generate", run_type="llm")
def generate_node(state: RetrievalState) -> dict:
    prompt = state["prompt"]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,    # deterministic — RAG answers should not be creative
        max_tokens=1000,
    )

    answer = response.choices[0].message.content.strip()
    print(f"[generate] Answer generated ({len(answer)} chars)")

    return {"answer": answer}
