"""
Chunking configuration.

Change CHUNKING_STRATEGY to switch between experiments.
Each run is traced separately in LangSmith under the project name.
"""

# ---------------------------------------------------------------
# Active strategy: "fixed" | "structure" | "semantic" | "sentence_window" | "parent_child"
# ---------------------------------------------------------------
CHUNKING_STRATEGY = "parent_child"

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
# Semantic chunking
# Splits by semantic similarity between consecutive sentences.
# A new chunk starts when cosine similarity drops below the threshold.
# ---------------------------------------------------------------
SEMANTIC = {
    "min_chunk_chars": 100,      # minimum chunk size — avoids tiny chunks
    "max_chunk_chars": 2000,     # fallback split if a semantic section is too long
    "similarity_threshold": 0.5, # lower = more splits, higher = fewer splits
}

# ---------------------------------------------------------------
# Sentence window chunking
# Indexes individual sentences but includes surrounding context.
# window_size = number of sentences on each side of the indexed sentence.
# ---------------------------------------------------------------
SENTENCE_WINDOW = {
    "window_size": 2,            # sentences of context on each side
    "min_sentence_chars": 30,    # ignore very short sentences (e.g. "Ok.", "Yes.")
}

# ---------------------------------------------------------------
# Parent-child chunking
# Children are small chunks indexed for precise retrieval.
# Parents are larger chunks passed to GPT-4o as context.
# ---------------------------------------------------------------
PARENT_CHILD = {
    "parent_chunk_size": 1500,   # parent chunk size in chars
    "parent_chunk_overlap": 100,
    "child_chunk_size": 300,     # child chunk size in chars
    "child_chunk_overlap": 30,
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
    "batch_size": 20,
}
