# ── Retrieval configuration ───────────────────────────────────────────────────
# Change these values to switch between strategies without touching pipeline code.

# Retrieval strategy:
#   "knn"    → pure semantic search using vector similarity
#   "hybrid" → combines kNN (semantic) + BM25 (keyword) for better coverage
RETRIEVAL_STRATEGY = "hybrid"  # "knn" | "hybrid"

# Weight for combining kNN and BM25 scores in hybrid search.
# 0.0 = pure BM25, 1.0 = pure kNN, 0.5 = equal weight (recommended starting point)
HYBRID_KNN_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3

# Number of candidate chunks to retrieve from OpenSearch.
# When reranking is enabled, this should be larger than TOP_K_FINAL
# so the reranker has enough candidates to work with.
RETRIEVAL_TOP_K_CANDIDATES = 20

# Number of chunks that reach the prompt after reranking.
# If reranking is disabled, the top RETRIEVAL_TOP_K_FINAL candidates are used directly.
RETRIEVAL_TOP_K_FINAL = 5

# ── Reranking ─────────────────────────────────────────────────────────────────
RERANKING_ENABLED = True

# Cohere reranking model.
# Options: "rerank-english-v3.0", "rerank-multilingual-v3.0", "rerank-english-v2.0"
RERANKING_MODEL = "rerank-english-v3.0"
