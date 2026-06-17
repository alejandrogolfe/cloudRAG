FROM python:3.11-slim

WORKDIR /opt/project

# Install curl for ECS health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install dependencies first — cached as a separate layer
# so rebuilds are fast when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY retrieval/ ./retrieval/
COPY config/ ./config/
COPY monitoring/ ./monitoring/

# Don't run as root
RUN useradd -m appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "retrieval.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
