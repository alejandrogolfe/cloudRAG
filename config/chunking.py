"""
Chunking configuration.

Change CHUNKING_STRATEGY to switch between experiments.
Each run is traced separately in LangSmith under the project name.
"""

# ---------------------------------------------------------------
# Active strategy: "fixed" | "structure"
# ---------------------------------------------------------------
CHUNKING_STRATEGY = "fixed"

# ---------------------------------------------------------------
# Fixed-size chunking (baseline)
# Splits by character count, no awareness of document structure.
# ---------------------------------------------------------------
FIXED_SIZE = {
    "chunk_size": 500,        # characters per chunk
    "chunk_overlap": 50,      # overlap between consecutive chunks
}

# ---------------------------------------------------------------
# Structure-aware chunking
# Splits by Markdown headers, recursive fallback if section too long.
# ---------------------------------------------------------------
STRUCTURE = {
    "max_chunk_chars": 1500,     # max chars before recursive fallback
    "chunk_overlap_chars": 200,  # overlap in recursive fallback
}

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
EVAL = {
    "queries_per_chunk": 2,   # synthetic queries generated per chunk
    "top_k": 5,               # how many chunks to retrieve per query
}

# ---------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------
EMBEDDING = {
    "model_id": "amazon.titan-embed-text-v2:0",
    "dimensions": 1024,
    "normalize": True,
    "batch_size": 10,
}
