# cloudRAG

A production-grade Retrieval-Augmented Generation (RAG) system built on AWS, using LangGraph for pipeline orchestration and LangSmith for observability.

---

## Architecture Overview

The system is split into two independent pipelines:

### Offline Pipeline (Ingestion)
Responsible for processing and indexing documents into the vector store. This pipeline runs **manually from local** whenever new documents need to be ingested. It does not require automation or continuous deployment.

```
Markdown / MDX Files
      ↓
  Load Node       → reads files recursively, extracts frontmatter + folder hierarchy as metadata
      ↓
  Filter Node     → discards noise (MediaWiki system pages, empty pages, navigation-only docs)
      ↓
  Clean Node      → removes boilerplate (redundant headers, navigation artifacts, horizontal rules)
      ↓
  Chunk Node      → fixed-size or structure-aware chunking (configurable via config/chunking.py)
      ↓
  Embed Node      → Amazon Bedrock Titan Embeddings v2 (1024 dimensions, normalized)
      ↓
  [saved to output/chunks_{strategy}.json]
      ↓
  run_upload.py   → upsert to Amazon OpenSearch Serverless
```

### Offline Evaluation
Before uploading to OpenSearch, the pipeline evaluates chunking quality locally using two metrics:

```
chunks_{strategy}.json
      ↓
  Synthetic dataset   → GPT-4o-mini generates (query, chunk_id) pairs per chunk
      ↓
  Local retrieval     → cosine similarity with numpy (no OpenSearch needed)
      ↓
  MRR + Hit Rate      → did the retriever find the right chunk?
      ↓
  Faithfulness        → does the retrieved chunk fully answer the query?
                        (detects cut/incomplete chunks that MRR cannot catch)
```

### Online Pipeline (Retrieval API)
A continuously running service that receives queries and returns generated answers. This pipeline runs on **AWS ECS** and is automatically updated on every merge to `main`.

```
User Query
      ↓
  Retrieve Node   → semantic search on OpenSearch
      ↓
  Augment Node    → context assembly
      ↓
  Generate Node   → OpenAI GPT-4o
      ↓
  Response
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Pipeline orchestration | LangGraph |
| Observability | LangSmith |
| Embeddings | Amazon Bedrock (Titan Embeddings v2) |
| Vector store | Amazon OpenSearch Serverless |
| LLM (online) | OpenAI GPT-4o |
| Evaluation | OpenAI GPT-4o-mini |
| Infrastructure | Terraform |
| Container registry | Amazon ECR |
| Container runtime | Amazon ECS (Fargate) |
| CI/CD | GitHub Actions |
| Local environment | Docker (PyCharm interpreter) |

---

## Project Structure

```
cloudRAG/
├── config/
│   └── chunking.py             # All chunking params + evaluation config (change strategy here)
├── ingestion/                  # Offline ingestion pipeline
│   ├── graph.py                # LangGraph graph: load → filter → clean → chunk → embed
│   ├── models.py               # Document and Chunk dataclasses
│   ├── state.py                # LangGraph shared state definition
│   └── nodes/
│       ├── loader.py           # Reads .md/.mdx recursively, extracts folder hierarchy
│       ├── filter.py           # Discards noisy/empty documents
│       ├── cleaner.py          # Removes MediaWiki boilerplate
│       ├── chunker.py          # Fixed-size and structure-aware strategies
│       └── embedder.py         # Bedrock Titan v2 embeddings with throttle protection
├── evaluation/
│   ├── dataset.py              # Synthetic query generation with GPT-4o-mini
│   ├── retriever.py            # Local cosine similarity search (numpy)
│   ├── metrics.py              # MRR and Hit Rate
│   └── faithfulness.py         # Chunk completeness evaluation with GPT-4o-mini
├── retrieval/                  # Online pipeline (to be built in Phase 2)
├── infra/                      # Terraform (to be built in Phase 2)
│   ├── dev.tfvars
│   └── prod.tfvars
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD for online pipeline only (Phase 3)
├── data/                       # Input markdown/mdx files (not committed)
├── output/                     # chunks_{strategy}.json + eval_{strategy}.json (not committed)
├── Dockerfile                  # Online pipeline only (Phase 2)
├── requirements.txt
├── .env                        # Local secrets (not committed)
├── run_local.py                # Runs ingestion + evaluation locally (no OpenSearch needed)
└── run_upload.py               # Uploads finalized chunks to OpenSearch
```

---

## Chunking Strategies

The chunking strategy is controlled by a single line in `config/chunking.py`:

```python
CHUNKING_STRATEGY = "fixed"   # "fixed" | "structure"
```

**fixed** — baseline. Splits by character count using `RecursiveCharacterTextSplitter`. No awareness of document structure. Used to establish a baseline MRR before adding complexity.

**structure** — splits by Markdown headers (`#`, `##`, `###`). Each section is a natural semantic unit. Applies recursive fallback if a section exceeds `max_chunk_chars`. Headers are stored as metadata (`header_1`, `header_2`, `header_3`) on each chunk.

The evaluation workflow is: run fixed → record MRR and Faithfulness → switch to structure → compare. Only add complexity when the numbers justify it.

---

## Evaluation Metrics

**MRR (Mean Reciprocal Rank)** measures retrieval quality. For each query, finds the rank of the correct chunk in the top-k results. Score = 1/rank. MRR = mean across all queries. Range: 0 to 1, higher is better.

**Hit Rate @ k** measures whether the correct chunk appears anywhere in the top-k results.

**Faithfulness** measures chunk completeness. For each (query, retrieved chunk) pair, GPT-4o-mini scores whether the chunk fully answers the question: 1.0 (complete), 0.5 (partial / possibly cut), 0.0 (irrelevant). This catches incomplete chunks that MRR cannot detect.

Interpreting results:

| MRR | Faithfulness | Conclusion |
|---|---|---|
| High | High | Chunking strategy is good → upload to OpenSearch |
| High | Low | Retriever works but chunks are cut → change strategy |
| Low | High | Chunks are good but retriever struggles → adjust top-k or embeddings |
| Low | Low | Both chunking and retrieval need work |

---

## LangSmith Projects

Offline and online pipelines use separate LangSmith projects:

| Pipeline | LangSmith Project |
|---|---|
| Offline (ingestion + evaluation) | `cloudRAG-offline` |
| Online (retrieval) | `cloudRAG-online` |

When working with large document sets, disable tracing to avoid payload size errors:
```env
LANGCHAIN_TRACING_V2=false
```

---

## Environment Variables

```env
# LangSmith
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true        # set to false for large runs
LANGCHAIN_PROJECT=cloudRAG-offline

# AWS (leave empty if using AWS CLI profile)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=eu-west-1

# OpenSearch (only needed for run_upload.py)
OPENSEARCH_ENDPOINT=             # e.g. xxxx.eu-west-1.aoss.amazonaws.com
OPENSEARCH_INDEX=cloudrag-docs

# OpenAI (evaluation: synthetic queries + faithfulness)
OPENAI_API_KEY=
```

In AWS (ECS), these variables are injected as Secrets Manager secrets — the application code does not change.

---

## Development Phases

### Phase 1 — Offline Pipeline (current)
1. Iterate on chunking strategies locally using `run_local.py`
2. Compare MRR and Faithfulness between `fixed` and `structure` strategies
3. Use Terraform to provision dev OpenSearch Serverless collection
4. Upload the best chunking result with `run_upload.py`

### Phase 2 — Online Pipeline + Infrastructure
1. Build the retrieval pipeline (LangGraph graph: retrieve → augment → generate)
2. Use Terraform to provision prod infrastructure (OpenSearch + ECR + ECS)
3. Validate the full RAG loop locally pointing to dev OpenSearch
4. Merge to `main` to trigger first deployment to ECS

### Phase 3 — CI/CD (online pipeline only)
On every merge to `main`, GitHub Actions will:
1. Build the Docker image
2. Push to Amazon ECR
3. Update the ECS task definition
4. Trigger a rolling ECS service update

The offline pipeline is never part of CI/CD — it always runs manually from local.

---

## Running the Offline Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env .env.local

# Add your docs to data/ (supports .md and .mdx, subdirectories preserved)

# Run ingestion + evaluation (default docs path: /opt/project/data/)
python run_local.py

# Or with a custom path
python run_local.py --docs-path ./data/langgraph/

# When satisfied with the chunking strategy, upload to OpenSearch
python run_upload.py --chunks-path ./output/chunks_structure.json
```

---

## Infrastructure

All infrastructure is managed with Terraform and applied manually from local:

```bash
cd infra/

# Dev (for local development and evaluation)
terraform apply -var-file=dev.tfvars

# Prod
terraform apply -var-file=prod.tfvars

# Tear down dev when not in use (saves cost)
terraform destroy -var-file=dev.tfvars
```
