# cloudRAG

A production-grade Retrieval-Augmented Generation (RAG) system built on AWS, using LangGraph for pipeline orchestration and LangSmith for observability.

---

## Architecture

![cloudRAG Architecture](images/architecture.png)

---

## Architecture Overview

The system is split into two independent pipelines:

### Offline Pipeline (Ingestion)
Responsible for processing and indexing documents into the vector store. This pipeline runs **manually from local** whenever new documents need to be ingested.

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
  Embed Node      → OpenAI text-embedding-3-small (1536 dims)
      ↓
  [saved to output/chunks_{strategy}.json]
      ↓
  run_upload.py   → upsert to Amazon OpenSearch Serverless
```

### Offline Evaluation
Before uploading to OpenSearch, the pipeline evaluates chunking quality locally:

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
```

### Online Pipeline (Retrieval API)
A continuously running service on AWS ECS, automatically updated on every push to `master`.

```
User Query
      ↓
  ALB             → public HTTPS endpoint
      ↓
  Auth + Rate Limit → X-API-Key header (FastAPI Security) + slowapi
                       (default 10/minute, configurable via RATE_LIMIT)
      ↓
  Retrieve Node   → kNN or Hybrid search (kNN + BM25 with RRF) on OpenSearch Serverless
                    configurable via config/retrieval.py (RETRIEVAL_STRATEGY)
                    retries (tenacity) on OpenAI embedding + OpenSearch calls
      ↓
  Rerank Node     → Cohere Rerank API — top-20 candidates → top-5 final
                    configurable via RERANKING_ENABLED
                    retries (tenacity), falls back to top-5 kNN if Cohere stays down
      ↓
  Augment Node    → context assembly with reranked chunks
                    trims least-relevant chunks if prompt exceeds 120k tokens (tiktoken)
      ↓
  Generate Node   → OpenAI GPT-4o, retries (tenacity) on transient API errors
      ↓
  Response: { answer, sources }
      ↓
  Cost tracking (background thread, doesn't block the response):
    fetch trace cost from LangSmith → publish QueryCount/LatencySeconds/CostPerQuery
    to CloudWatch (namespace cloudRAG/Costs)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Pipeline orchestration | LangGraph |
| Observability / tracing | LangSmith |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) — same model in dev and prod |
| Vector store | Amazon OpenSearch Serverless |
| LLM (online) | OpenAI GPT-4o |
| Evaluation | OpenAI GPT-4o-mini |
| API | FastAPI + uvicorn |
| Auth | API key (`X-API-Key` header) |
| Rate limiting | slowapi (default 10/minute, `RATE_LIMIT` env var) |
| Resilience | tenacity retries (retriever, reranker, generator) + Cohere→kNN fallback |
| Token budgeting | tiktoken — trims context if prompt exceeds 120k tokens |
| Cost & usage tracking | boto3 → CloudWatch custom metrics (`cloudRAG/Costs`) + LangSmith trace cost |
| Daily infra cost tracking | AWS Lambda + EventBridge (06:00 UTC) reading Cost Explorer |
| Monitoring | CloudWatch Dashboard, Alarms (SNS), Logs Insights |
| Testing | pytest (19 tests, mocked graph — runs in CI before every deploy) |
| Infrastructure | Terraform |
| Container registry | Amazon ECR |
| Container runtime | Amazon ECS (Fargate) |
| Load balancer | AWS ALB |
| Secrets | AWS Secrets Manager |
| CI/CD | GitHub Actions |
| Local environment | Docker (PyCharm interpreter) |

---

## Project Structure

```
cloudRAG/
├── config/
│   ├── chunking.py             # Chunking strategy + evaluation config
│   └── retrieval.py            # Retrieval strategy (knn/hybrid), reranking, top-k
├── ingestion/                  # Offline ingestion pipeline
│   ├── graph.py                # LangGraph graph: load → filter → clean → chunk → embed
│   ├── models.py               # Document and Chunk dataclasses
│   ├── state.py                # LangGraph shared state definition
│   └── nodes/
│       ├── loader.py           # Reads .md/.mdx recursively, extracts folder hierarchy
│       ├── filter.py           # Discards noisy/empty documents
│       ├── cleaner.py          # Removes MediaWiki boilerplate
│       ├── chunker.py          # fixed, structure, semantic, sentence_window, parent_child
│       └── embedder.py         # Embeddings with throttle protection
├── evaluation/
│   ├── dataset.py              # Synthetic query generation with GPT-4o-mini
│   ├── retriever.py            # Local cosine similarity search (numpy)
│   ├── metrics.py              # MRR and Hit Rate
│   ├── faithfulness.py         # Chunk completeness evaluation with GPT-4o-mini
│   └── ragas_eval.py           # End-to-end pipeline evaluation with RAGAS
├── retrieval/                  # Online pipeline
│   ├── api.py                  # FastAPI app — POST /query, GET /health, auth + rate limiting, JSON logging
│   ├── graph.py                # LangGraph graph: retrieve → rerank → augment → generate
│   ├── models.py               # QueryRequest, QueryResponse, RetrievedChunk
│   ├── state.py                # RetrievalState
│   └── nodes/
│       ├── retriever.py        # kNN or hybrid search (RRF) on OpenSearch, with retries
│       ├── reranker.py         # Cohere Rerank API — top-20 → top-5, retries + kNN fallback
│       ├── augmenter.py        # Prompt assembly with reranked chunks, token-budget trimming
│       └── generator.py        # GPT-4o generation, with retries
├── monitoring/
│   └── cost_tracking.py        # Per-query LangSmith cost lookup + CloudWatch metric publishing
├── cost_lambda/
│   └── handler.py              # Daily Lambda: Cost Explorer → InfraCostDaily CloudWatch metric
├── tests/
│   ├── conftest.py             # Test fixtures — mocked graph, fake env vars (no real API keys needed)
│   ├── test_api.py             # API integration tests (auth, rate limiting, endpoints)
│   └── test_nodes.py           # Unit tests for retrieval nodes
├── infra/                      # Terraform — all infrastructure as code
│   ├── main.tf                 # Provider config and default tags
│   ├── variables.tf            # Input variables
│   ├── opensearch.tf           # OpenSearch Serverless + IAM role
│   ├── ecr.tf                  # ECR repository + lifecycle policy
│   ├── ecs.tf                  # ECS cluster + task definition + service + log group
│   ├── networking.tf           # VPC + subnets + security groups + ALB
│   ├── cost_lambda.tf           # Cost-tracker Lambda + EventBridge daily schedule + IAM
│   ├── dashboard.tf            # CloudWatch Dashboard (cost/latency/volume by config, top users)
│   ├── alarms.tf                # SNS topic + 3 CloudWatch alarms (cost/query, daily budget, anomaly)
│   ├── outputs.tf              # Outputs: endpoints, URLs, ARNs
│   ├── dev.tfvars              # Dev environment values (not committed)
│   └── prod.tfvars             # Prod environment values (not committed)
├── .github/
│   └── workflows/
│       ├── deploy.yml             # CI/CD: tests → build → ECR → ECS on every push to master
│       └── deploy-cost-lambda.yml # CI/CD: redeploys cost_lambda/handler.py on change
├── data/                       # Input markdown/mdx files (not committed)
├── output/                     # chunks_{strategy}.json + eval_{strategy}.json (not committed)
├── Dockerfile                  # Online pipeline — starts uvicorn, HEALTHCHECK for ECS
├── requirements.txt
├── requirements-test.txt       # Test dependencies (pytest, httpx, etc.)
├── requirements-eval.txt       # Separate dependencies for RAGAS evaluation (isolated venv)
├── .env                        # Local secrets (not committed)
├── run_local.py                # Ingestion + evaluation locally (no OpenSearch needed)
├── run_upload.py               # Uploads chunks to OpenSearch Serverless
├── verify_upload.py            # Verifies index count, embeddings, and kNN query
├── delete_index.py             # Deletes the OpenSearch index (use when recreating mapping)
├── create_secrets.py           # Pushes .env values to AWS Secrets Manager (incl. API_KEY)
├── deploy_image.py             # Builds and pushes Docker image to ECR manually
├── evaluation/
│   ├── create_langsmith_dataset.py # Creates synthetic dataset in LangSmith
│   └── run_langsmith_eval.py       # Runs evaluation experiment against LangSmith dataset
```

---

## Deploying from Scratch

Run these steps in order every time you recreate the infrastructure.

### Prerequisites
- AWS CLI configured (`aws sts get-caller-identity` works)
- Terraform installed
- Docker running
- Python environment with `pip install -r requirements.txt`
- GitHub secrets configured (see CI/CD section below)

### Step 1 — Fill in dev.tfvars

```bash
aws sts get-caller-identity --query Arn --output text
```

Edit `infra/dev.tfvars` and set `admin_iam_principal_arn` to your ARN. Also set `alert_email` (required, no default) — CloudWatch alarms send notifications there via SNS.

### Step 2 — Deploy infrastructure

```bash
cd infra/
terraform init
terraform apply -var-file="dev.tfvars"
```

Creates: OpenSearch Serverless, ECR, ECS cluster, VPC, ALB, IAM roles, CloudWatch logs, the cost-tracker Lambda + daily EventBridge schedule, the CloudWatch Dashboard, and 3 CloudWatch alarms + SNS topic.

AWS sends a confirmation email to `alert_email` for the SNS subscription — click the link or alarms won't notify you.

Copy the outputs:

```bash
terraform output opensearch_endpoint   # update OPENSEARCH_ENDPOINT in .env (no https://)
terraform output alb_url               # public API URL
```

### Step 3 — Update .env

```env
OPENSEARCH_ENDPOINT=xxxx.eu-west-1.aoss.amazonaws.com
OPENSEARCH_INDEX=cloudrag-docs
AWS_REGION=eu-west-1
API_KEY=                # value clients must send as X-API-Key
COHERE_API_KEY=          # required if RERANKING_ENABLED = True
```

### Step 4 — Push secrets to AWS Secrets Manager

```bash
# Run from local (not Docker) — needs AWS CLI credentials
python create_secrets.py
```

This pushes every key in `KEYS_TO_PUSH` (`create_secrets.py`), including `API_KEY`, to `cloudrag/*` secrets — Terraform's ECS task definition already references all of them.

### Step 5 — Upload chunks to OpenSearch

```bash
python run_upload.py --chunks-path ./output/chunks_fixed.json
python verify_upload.py --index cloudrag-docs --chunks-path ./output/chunks_fixed.json
```

If `chunks_fixed.json` doesn't exist yet:

```bash
python run_local.py --docs-path ./data/
```

### Step 6 — Build and push Docker image to ECR

```bash
# Run from local (not Docker) — only needed for the very first deploy.
# After that, every push to master rebuilds and redeploys via CI/CD.
python deploy_image.py
```

### Step 7 — Verify the API

```bash
curl http://<alb_url>/health
# → {"status": "ok"}

curl -X POST http://<alb_url>/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your API_KEY value>" \
  -d '{"question": "What is context engineering?"}'
```

Cost and latency per query show up within ~1-2 minutes at **CloudWatch → Dashboards → `cloudRAG-<environment>`**. The daily infra-cost panel needs the cost-tracker Lambda's first run (06:00 UTC) to populate.

### Tear down when not in use

```bash
cd infra/
terraform destroy -var-file="dev.tfvars"
```

This destroys everything Terraform tracks — including the OpenSearch collection (all indexed chunks are lost; re-run ingestion + upload after recreating). Cost when running: ~$0.24/hour (OpenSearch) + ECS Fargate usage.

---

## CI/CD

There are two independent workflows:

### `deploy.yml` — application deploy

On every push to `master` that modifies `retrieval/`, `config/`, `ingestion/`, `tests/`, `Dockerfile`, `requirements.txt`, or `requirements-test.txt`, GitHub Actions automatically:

1. Runs the test suite (`pytest tests/`) — mocked graph, no real API keys needed, deploy is blocked if tests fail
2. Builds the Docker image
3. Pushes to ECR (tagged with commit SHA and `latest`)
4. Triggers a rolling ECS deployment (`--force-new-deployment`)
5. Waits for the service to stabilize

### `deploy-cost-lambda.yml` — cost-tracker Lambda deploy

On every push to `master` that modifies `cost_lambda/` (or manually via `workflow_dispatch`):

1. Zips `cost_lambda/handler.py`
2. Calls `aws lambda update-function-code`
3. Waits for the update to complete

Terraform creates the Lambda on first `apply`; this workflow only updates its code afterwards (`infra/cost_lambda.tf` has `ignore_changes = [filename, source_code_hash]` so a later `terraform apply` doesn't revert a CI/CD deploy).

**Required GitHub secrets** (Settings → Secrets and variables → Actions):

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |

`AWS_REGION` and the ECR/ECS/Lambda names are hardcoded as workflow `env` values (not secrets) — update them there if you rename resources. Runtime secrets (`OPENAI_API_KEY`, `COHERE_API_KEY`, `LANGCHAIN_*`, `API_KEY`, etc.) live in AWS Secrets Manager (pushed via `create_secrets.py`), not in GitHub — the CI/CD pipeline never needs them since tests run fully mocked.

The offline pipeline is never part of CI/CD — it always runs manually from local.

---

## Testing the API locally

```bash
docker run --rm -p 8000:8000 --env-file .env \
  -v ${PWD}:/opt/project -w /opt/project \
  rag_python:latest \
  uvicorn retrieval.api:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive FastAPI UI — click **Authorize** and enter your `API_KEY` value to call `/query` from the browser.

---

## Retrieval Configuration

Controlled by `config/retrieval.py`:

```python
RETRIEVAL_STRATEGY = "knn"      # "knn" | "hybrid"
HYBRID_KNN_WEIGHT  = 0.7        # weight for kNN in hybrid fusion
HYBRID_BM25_WEIGHT = 0.3        # weight for BM25 in hybrid fusion
RETRIEVAL_TOP_K_CANDIDATES = 20 # candidates retrieved from OpenSearch
RETRIEVAL_TOP_K_FINAL = 5       # chunks passed to the prompt after reranking
RERANKING_ENABLED = True        # enable/disable Cohere reranking
RERANKING_MODEL = "rerank-english-v3.0"
```

**knn** — pure semantic search using vector similarity. Fast, works well for conceptual queries.

**hybrid** — combines kNN and BM25 using Reciprocal Rank Fusion (RRF). Better coverage for queries with exact technical terms (function names, versions, IDs). Implemented in Python since OpenSearch Serverless does not support search pipelines.

**Reranking** — Cohere cross-encoder reorders the top-20 candidates by relevance before passing top-5 to GPT-4o. Improves precision without changing the index.

---

## Chunking Strategies

Controlled by `config/chunking.py`:

```python
CHUNKING_STRATEGY = "fixed"   # "fixed" | "structure" | "semantic" | "sentence_window" | "parent_child"
```

**fixed** — splits by character count. Baseline, no structure awareness.

**structure** — splits by Markdown headers. Each section is a semantic unit. Headers stored as metadata. Best overall for structured documentation.

**semantic** — splits by semantic similarity between consecutive sentences using embeddings. Detects topic changes automatically. Slower to generate (requires embeddings during ingestion).

**sentence_window** — indexes individual sentences but passes surrounding context to GPT-4o. High MRR due to precise matching, but many chunks are incomplete sentences.

**parent_child** — indexes small child chunks for precise retrieval, but passes the larger parent chunk as context to GPT-4o. Offline metrics underestimate quality — the real benefit shows at generation time.

Workflow: run fixed → record MRR and Faithfulness → switch to structure → compare. Only add complexity when numbers justify it.

**Evaluating chunking strategies:**

```bash
# Run ingestion + evaluation
python run_local.py --docs-path ./data/

# Re-evaluate without regenerating chunks (faster, uses existing output/chunks_{strategy}.json)
python run_local.py --skip-ingestion
```

---

## Evaluation

The system has two independent evaluation layers — one for chunking quality and one for the full pipeline.

---

### Layer 1 — Chunking Evaluation (offline, no API needed)

Runs locally against the generated chunks without needing OpenSearch or the API. Use this to compare chunking strategies before uploading to OpenSearch.

```bash
# Generate chunks and evaluate
python run_local.py --docs-path ./data/

# Re-evaluate existing chunks without regenerating (faster)
python run_local.py --skip-ingestion
```

Results are saved to `output/eval_{strategy}.json`.

**Metrics:**

**MRR (Mean Reciprocal Rank)** — measures retrieval quality. For each synthetic query, finds the rank of the correct chunk in the top-k results. Score = 1/rank. Range 0-1, higher is better.

**Hit Rate @ k** — whether the correct chunk appears anywhere in the top-k results.

**Faithfulness** — chunk completeness. GPT-4o-mini scores whether each chunk fully answers its query. Catches cut/incomplete chunks that MRR cannot detect.

| MRR | Faithfulness | Conclusion |
|---|---|---|
| High | High | Good → upload to OpenSearch |
| High | Low | Chunks are cut → change strategy |
| Low | High | Retriever struggles → adjust top-k or embeddings |
| Low | Low | Both need work |

**Chunking strategy comparison results:**

| Strategy | MRR | Hit Rate @5 | Faithfulness | Notes |
|---|---|---|---|---|
| fixed | 0.703 | 0.878 | 0.661 | Baseline |
| structure | 0.732 | 0.921 | 0.724 | Best overall — respects document structure |
| sentence_window | 0.835 | 0.945 | 0.716 | High MRR but many cut chunks |
| parent_child | 0.694 | 0.847 | 0.577 | Offline metrics underestimate — context comes from parent at runtime |

**Current strategy in production: `structure`**

---

### Layer 2 — Pipeline Evaluation (end-to-end, requires API)

Evaluates the full pipeline — retrieval + reranking + generation — using RAGAS. Requires the API to be running (locally or on AWS).

RAGAS has dependency conflicts with the main project — run it in a separate virtual environment:

```bash
# Create and activate the evaluation venv (first time only)
python -m venv .venv-eval
.venv-eval\Scripts\activate        # Windows
# source .venv-eval/bin/activate   # Mac/Linux
pip install -r requirements-eval.txt

# Against local API (activate venv first)
python evaluation/ragas_eval.py --api-url http://localhost:8000 --chunks-path ./output/chunks_structure.json --max-samples 20

# Against deployed API
python evaluation/ragas_eval.py --api-url http://<alb_url> --chunks-path ./output/chunks_structure.json --max-samples 50
```

Results are saved to `output/ragas_eval.json`.

**Metrics:**

**context_precision** — of the chunks retrieved, how many are actually relevant to the question? Low value means the retriever or reranker is returning noisy results.

**context_recall** — were all the necessary chunks retrieved to answer the question? Low value means relevant information exists in the index but was not retrieved.

**faithfulness** — is the generated answer grounded in the retrieved context, or did GPT-4o hallucinate? The most important metric for production safety.

**answer_relevancy** — does the answer actually address the question asked? A response can be faithful but still not answer the question.

**When to run each evaluation:**

| You changed | Run |
|---|---|
| Chunking strategy | `run_local.py` → MRR + Faithfulness |
| Retrieval strategy (knn/hybrid) | `ragas_eval.py` → context_precision + context_recall |
| Reranking (on/off, model) | `ragas_eval.py` → context_precision |
| Augmenter prompt | `ragas_eval.py` → faithfulness + answer_relevancy |
| Generator model | `ragas_eval.py` → faithfulness + answer_relevancy |

---

### Layer 3 — LangSmith Evaluation (end-to-end, experiment tracking)

Evaluates the full pipeline against a persistent dataset stored in LangSmith.
Each run creates a named experiment with the full configuration as metadata,
allowing comparison across different configurations (chunking, retrieval, reranking).

```bash
# Create the dataset in LangSmith (first time only)
# Run from the evaluation venv
python evaluation/create_langsmith_dataset.py --chunks-path ./output/chunks_structure.json --max-chunks 25

# Run an evaluation experiment
python evaluation/run_langsmith_eval.py --api-url http://localhost:8000
```

Experiments are named automatically from the current config, e.g.:
`structure-hybrid-rerank-20260611-1030`

**Metrics:**

**answer_correctness** — is the generated answer factually correct compared to the ground truth?

**hallucination** — does the answer contain only information present in the retrieved context?

**document_relevance** — are the retrieved documents relevant to the question?

Results are visible at [smith.langchain.com](https://smith.langchain.com) under Datasets & Experiments.

The dataset grows over time — add real user queries from the LangSmith dashboard (Tracing → select a run → Add to dataset).

---

### Observability — LangSmith Tracing

Every query hitting the API is automatically traced in LangSmith thanks to `@traceable` decorators on each node.

- **Local**: traces go to project `cloudRAG-offline`
- **ECS**: traces go to project `cloudRAG-online`

Each trace shows the full pipeline waterfall: retrieve → rerank → augment → generate, with latency, inputs and outputs per node, token usage and cost.

Note: set `LANGCHAIN_TRACING_V2=true` (lowercase) in `.env` — uppercase `True` is not recognized.

---

### Cost Tracking & Monitoring

Two layers of CloudWatch metrics, both in the `cloudRAG/Costs` namespace:

**Per-query (real-time)** — `monitoring/cost_tracking.py` runs in a background thread after every `/query` response (doesn't add latency). It looks up the query's LLM cost from its LangSmith trace (retrying a few times since LangSmith computes cost asynchronously) and publishes `QueryCount`, `LatencySeconds`, `CostPerQuery` — once with dimensions (`ConfigName`, `Model`) for the dashboard, once without dimensions for alarms (CloudWatch alarms don't support `SEARCH()`). `UserId` is intentionally **not** a CloudWatch dimension — high cardinality would blow up custom-metric cost — it's logged instead as a structured field and queried via Logs Insights.

**Daily infra cost** — `cost_lambda/handler.py`, triggered by EventBridge at 06:00 UTC, reads AWS Cost Explorer for the previous day (filtered by the `Project=cloudRAG` tag, grouped by service) and publishes `InfraCostDaily`.

**Dashboard** (`infra/dashboard.tf`, CloudWatch → Dashboards → `cloudRAG-<environment>`): avg cost/query by config, daily total LLM cost, cost by model, query volume by config, avg latency by config, combined infra+LLM cost, and two Logs Insights widgets (top users by cost / by query volume, parsed from the structured log line).

**Alarms** (`infra/alarms.tf`, notify via SNS to `alert_email`):
- `cost_per_query` — average cost per query exceeds `cost_per_query_threshold_usd` (default $0.01)
- `daily_cost_budget` — total daily LLM cost exceeds `daily_cost_budget_usd` (default $5)
- `cost_anomaly` — `CostPerQuery` deviates from its expected band (`anomaly_detection_std_devs`, default 2σ)

---

## Environment Variables

```env
# LangSmith
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=cloudRAG-offline   # use cloudRAG-online for ECS

# AWS
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=eu-west-1

# OpenSearch
OPENSEARCH_ENDPOINT=    # no https://, e.g. xxxx.eu-west-1.aoss.amazonaws.com
OPENSEARCH_INDEX=cloudrag-docs

# OpenAI
OPENAI_API_KEY=

# Cohere (reranking)
COHERE_API_KEY=

# API
API_KEY=                # required — value clients must send as X-API-Key
RATE_LIMIT=10/minute     # optional, defaults to 10/minute
```

Note on vector dimensions: embeddings use OpenAI text-embedding-3-small (1536 dims) in both dev and prod. If you switch to a different embedding model/provider, update `VECTOR_DIMENSION` in `run_upload.py` (and the index mapping) to match.

In ECS, all variables are injected from AWS Secrets Manager — the application code does not change.
