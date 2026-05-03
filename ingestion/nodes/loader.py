import os
import re
from typing import Dict, Tuple
from ingestion.state import IngestionState
from ingestion.models import Document


def _parse_frontmatter(content: str) -> Tuple[Dict, str]:
    frontmatter = {}
    body = content
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if match:
        raw = match.group(1)
        body = content[match.end():]
        for line in raw.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                frontmatter[key.strip()] = value.strip()
    return frontmatter, body


def _get_sections(docs_path: str, filepath: str) -> Dict:
    """
    Extracts folder hierarchy as section metadata.
    Example: data/langgraph/concepts/low_level.mdx
      -> section_1: langgraph
      -> section_2: concepts
      -> section_3: None
    """
    rel_path = os.path.relpath(filepath, docs_path)
    parts = rel_path.replace("\\", "/").split("/")
    folders = parts[:-1]  # everything except the filename
    return {
        "section_1": folders[0] if len(folders) > 0 else None,
        "section_2": folders[1] if len(folders) > 1 else None,
        "section_3": folders[2] if len(folders) > 2 else None,
    }


def load_node(state: IngestionState) -> dict:
    docs_path = state["docs_path"]
    documents = []

    # os.walk recurses into all subdirectories
    for root, _, files in os.walk(docs_path):
        for filename in files:
            if not filename.endswith((".md", ".mdx")):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_content = f.read()
            except Exception as e:
                print(f"[load] ERROR reading {filepath}: {e}")
                continue

            frontmatter, body = _parse_frontmatter(raw_content)
            sections = _get_sections(docs_path, filepath)

            documents.append(Document(
                source=filepath,
                title=frontmatter.get("title", filename),
                url=frontmatter.get("url", ""),
                content=body,
                metadata={
                    "filename": filename,
                    **sections,
                }
            ))

    print(f"[load] {len(documents)} documents loaded")
    return {"raw_documents": documents}
