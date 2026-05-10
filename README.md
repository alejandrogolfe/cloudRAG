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
  Embed Node      → OpenAI text-embedding-3-small (1536 dims, local dev)
                    Amazon Bedrock Titan Embeddings v2 (1024 dims, production)
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
| Embeddings (dev) | OpenAI text-embedding-3-small (1536 dims) |
| Embeddings (prod) | Amazon Bedrock Titan Embeddings v2 (1024 dims) |
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
│       └── embedder.py         # Embeddings with throttle protection
├── evaluation/
│   ├── dataset.py              # Synthetic query generation with GPT-4o-mini
│   ├── retriever.py            # Local cosine similarity search (numpy)
│   ├── metrics.py              # MRR and Hit Rate
│   └── faithfulness.py         # Chunk completeness evaluation with GPT-4o-mini
├── retrieval/                  # Online pipeline (Phase 3)
├── infra/                      # Terraform — all infrastructure as code
│   ├── main.tf                 # Provider config and default tags
│   ├── variables.tf            # Input variables (region, environment, ARNs)
│   ├── opensearch.tf           # OpenSearch Serverless: collection + 3 required policies + IAM role
│   ├── outputs.tf              # Outputs: endpoint, collection ARN, app role ARN
│   ├── dev.tfvars              # Dev environment values (not committed)
│   └── prod.tfvars             # Prod environment values (not committed)
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD for online pipeline only (Phase 3)
├── data/                       # Input markdown/mdx files (not committed)
├── output/                     # chunks_{strategy}.json + eval_{strategy}.json (not committed)
├── Dockerfile                  # Online pipeline only (Phase 3)
├── requirements.txt
├── .env                        # Local secrets (not committed)
├── run_local.py                # Runs ingestion + evaluation locally (no OpenSearch needed)
├── run_upload.py               # Uploads finalized chunks to OpenSearch Serverless
├── verify_upload.py            # Verifies index count, embeddings, and kNN query after upload
└── delete_index.py             # Deletes the OpenSearch index (use when recreating with new mapping)
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

# OpenSearch (only needed for run_upload.py and verify_upload.py)
OPENSEARCH_ENDPOINT=             # e.g. xxxx.eu-west-1.aoss.amazonaws.com (no https://)
OPENSEARCH_INDEX=cloudrag-docs

# OpenAI (embeddings in dev + evaluation)
OPENAI_API_KEY=
```

**Note on embeddings and vector dimensions:**
- Local dev uses `text-embedding-3-small` (1536 dims) — `VECTOR_DIMENSION = 1536` in `run_upload.py`
- Production uses Amazon Bedrock Titan Embeddings v2 (1024 dims) — update `VECTOR_DIMENSION = 1024` when switching

In AWS (ECS), these variables are injected as Secrets Manager secrets — the application code does not change.

---

## Development Phases

### ✅ Phase 1 — Offline Pipeline (complete)
1. Ingestion pipeline: load → filter → clean → chunk → embed
2. Evaluation: MRR + Hit Rate + Faithfulness comparison between `fixed` and `structure` strategies
3. Infrastructure: OpenSearch Serverless provisioned with Terraform (`infra/`)
4. Upload: chunks indexed to OpenSearch with `run_upload.py`, verified with `verify_upload.py`

### Phase 2 — Online Pipeline (next)
1. Build the retrieval pipeline (LangGraph graph: retrieve → augment → generate)
2. Provision prod infrastructure: ECR + ECS with Terraform
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
cp .env.example .env

# Add your docs to data/ (supports .md and .mdx, subdirectories preserved)

# Run ingestion + evaluation (default docs path: /opt/project/data/)
python run_local.py

# Or with a custom path
python run_local.py --docs-path ./data/langgraph/

# When satisfied with the chunking strategy, upload to OpenSearch
python run_upload.py --chunks-path ./output/chunks_fixed.json

# Verify the upload
python verify_upload.py --index cloudrag-docs --chunks-path ./output/chunks_fixed.json
```

---

## Infrastructure

All infrastructure is managed with Terraform and applied manually from local.

**First time setup — find your IAM ARN:**
```bash
aws sts get-caller-identity --query Arn --output text
# Copy the output into dev.tfvars as admin_iam_principal_arn
```

**Deploy / destroy:**
```bash
cd infra/

# Deploy dev (Windows)
terraform apply -var-file="dev.tfvars"

# Deploy dev (Mac/Linux)
terraform apply -var-file=dev.tfvars

# Get the OpenSearch endpoint after apply — copy to .env
terraform output opensearch_endpoint

# Tear down dev when not in use (saves ~$0.24/hour)
terraform destroy -var-file="dev.tfvars"
```

**Resuming after destroy — full workflow:**
```bash
# 1. Recreate infrastructure
cd infra && terraform apply -var-file="dev.tfvars"

# 2. Update OPENSEARCH_ENDPOINT in .env with new endpoint
terraform output opensearch_endpoint

# 3. Re-upload chunks (chunks_fixed.json already exists locally)
cd .. && python run_upload.py --chunks-path ./output/chunks_fixed.json

# 4. Verify
python verify_upload.py --index cloudrag-docs --chunks-path ./output/chunks_fixed.json
```

**OpenSearch Serverless requires three policy types** (all managed by Terraform):
- Encryption policy — KMS key for data at rest
- Network policy — public HTTPS access (requests still require SigV4 signatures)
- Data access policy — index-level permissions per IAM principal

**Note:** `dev.tfvars` and `prod.tfvars` are excluded from git (`.gitignore`) because they contain your AWS account ID and IAM ARN.
