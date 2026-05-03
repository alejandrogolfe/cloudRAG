import re
from ingestion.state import IngestionState
from ingestion.models import Document


def _clean(content: str) -> str:
    content = re.sub(r"^# .+\n", "", content, count=1)
    content = re.sub(r"^\*\*URL:\*\*.+\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"^---\n", "", content, flags=re.MULTILINE)
    content = re.sub(r'Retrieved from ".+?".*$', "", content, flags=re.DOTALL)
    for line in ["Jump to navigation", "Jump to search", "Navigation menu", "From Wiki CVBLab"]:
        content = content.replace(line, "")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def clean_node(state: IngestionState) -> dict:
    cleaned = []
    for doc in state["filtered_documents"]:
        cleaned.append(Document(
            source=doc.source,
            title=doc.title,
            url=doc.url,
            content=_clean(doc.content),
            metadata=doc.metadata,
        ))
    print(f"[clean] {len(cleaned)} documents cleaned")
    return {"cleaned_documents": cleaned}
