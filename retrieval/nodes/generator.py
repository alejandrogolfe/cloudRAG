"""
Generator node: calls GPT-4o with the assembled prompt and returns the answer.

Uses langchain_openai instead of openai directly so LangSmith
automatically captures token usage and cost per query.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from retrieval.state import RetrievalState
from langsmith import traceable

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1000,
)


@traceable(name="generate", run_type="llm")
def generate_node(state: RetrievalState) -> dict:
    prompt = state["prompt"]

    response = llm.invoke([HumanMessage(content=prompt)])

    answer = response.content.strip()
    print(f"[generate] Answer generated ({len(answer)} chars)")

    return {"answer": answer}
