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
# ---------------------------------------------------------------
FIXED_SIZE = {
    "chunk_size": 500,
    "chunk_overlap": 50,
}

# ---------------------------------------------------------------
# Structure-aware chunking
# ---------------------------------------------------------------
STRUCTURE = {
    "max_chunk_chars": 1500,
    "chunk_overlap_chars": 200,
}

# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
EVAL = {
    "queries_per_chunk": 2,
    "top_k": 5,
}

# ---------------------------------------------------------------
# Embedding
# NOTE: using OpenAI for local dev, swap to Bedrock for production
#
# OpenAI text-embedding-3-small: 1536 dimensions
# Bedrock Titan v2:               1024 dimensions (change accordingly)
# ---------------------------------------------------------------
EMBEDDING = {
    "model_id": "text-embedding-3-small",   # OpenAI model
    "dimensions": 1536,
    "batch_size": 20,                        # OpenAI supports larger batches
}
