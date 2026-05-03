from typing import List
from ingestion.state import IngestionState
from ingestion.models import Document

NOISY_TITLE_PATTERNS = ["Category:", "File:", "Special:", "Help:", "Talk:", "Revision history"]
NOISY_CONTENT_PATTERNS = [
    "There is currently no text in this page",
    "Revision history",
    "Filter revisions",
    "Jump to navigation",
    "diff) ← Older revision",
]
MIN_CONTENT_LENGTH = 200


def _is_noisy(doc: Document) -> bool:
    for p in NOISY_TITLE_PATTERNS:
        if p in doc.title:
            return True
    for p in NOISY_CONTENT_PATTERNS:
        if p in doc.content:
            return True
    if len(doc.content.strip()) < MIN_CONTENT_LENGTH:
        return True
    return False


def filter_node(state: IngestionState) -> dict:
    filtered, discarded = [], []
    for doc in state["raw_documents"]:
        if _is_noisy(doc):
            discarded.append(doc)
        else:
            filtered.append(doc)

    print(f"[filter] {len(filtered)} kept, {len(discarded)} discarded")
    return {"filtered_documents": filtered, "filtered_out": discarded}
