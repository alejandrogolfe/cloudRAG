FROM python:3.11-slim

WORKDIR /opt/project

# Install dependencies first — cached as a separate layer
# so rebuilds are fast when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY retrieval/ ./retrieval/
COPY config/ ./config/

# Don't run as root
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Start the FastAPI app with uvicorn
# Workers=1 for dev — increase in prod via ECS task definition env var
CMD ["uvicorn", "retrieval.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
