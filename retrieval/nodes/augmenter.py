"""
Augmenter node: assembles the prompt from the retrieved chunks.

The prompt structure follows a standard RAG pattern:
- System: role + instructions
- User: question + context chunks

Keeping the prompt assembly in its own node makes it easy to iterate
on prompt engineering without touching retrieval or generation logic.
"""

from retrieval.state import RetrievalState


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer the question using only the information in the context below.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and direct.
- Do not make up information that is not in the context."""


def augment_node(state: RetrievalState) -> dict:
    question = state["question"]
    chunks = state["reranked_chunks"]

    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = " > ".join(filter(None, [chunk.header_1, chunk.header_2, chunk.header_3]))
        label = f"[{i}] {chunk.title}"
        if header:
            label += f" — {header}"
        context_parts.append(f"{label}\n{chunk.content}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"{SYSTEM_PROMPT}\n\n## Context\n\n{context}\n\n## Question\n\n{question}"

    print(f"[augment] Prompt built — {len(chunks)} chunks, {len(prompt)} chars")

    return {"prompt": prompt}
