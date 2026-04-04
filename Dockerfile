# ── ATC TRACON RL Environment — Docker Image ──────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="ATC RL Team"
LABEL description="OpenEnv-compliant ATC TRACON RL Environment"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models.py        /app/models.py
COPY server/your_environment.py /app/your_environment.py
COPY server/app.py    /app/app.py

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Env vars (override at runtime)
ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=claude-sonnet-4-20250514
ENV HF_TOKEN=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]