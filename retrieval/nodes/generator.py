"""
Generator node: calls GPT-4o with the assembled prompt and returns the answer.

Uses langchain_openai instead of openai directly so LangSmith
automatically captures token usage and cost per query.
"""

import logging
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from retrieval.state import RetrievalState
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1000,
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )),
    reraise=True,
)
def _call_llm(prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


@traceable(name="generate", run_type="llm")
def generate_node(state: RetrievalState) -> dict:
    answer = _call_llm(state["prompt"])
    logger.info(f"generate — answer generated ({len(answer)} chars)")
    return {"answer": answer}
