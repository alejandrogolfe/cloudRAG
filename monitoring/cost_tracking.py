import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_trace_cost(run_id: str, max_attempts: int = 3, wait_seconds: float = 1.5) -> Optional[float]:
    """Fetch the cost of a specific LangSmith trace with retry backoff.

    LangSmith calculates cost asynchronously after the run completes, so we
    retry a few times before giving up. Returns None if tracing is disabled,
    the cost hasn't arrived in time, or the API call fails.
    """
    if os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() != "true":
        return None

    try:
        from langsmith import Client
        client = Client()
        for attempt in range(max_attempts):
            run = client.read_run(run_id)
            if run.total_cost is not None:
                return float(run.total_cost)
            if attempt < max_attempts - 1:
                time.sleep(wait_seconds)
        logger.warning("LangSmith cost not yet available for run %s after %d attempts", run_id, max_attempts)
        return None
    except Exception as e:
        logger.warning("Failed to read LangSmith run %s: %s", run_id, e)
        return None
