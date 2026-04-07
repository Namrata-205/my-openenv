# ── ATC TRACON RL Environment — Dockerfile ────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenEnv CLI
RUN pip install --no-cache-dir openenv-core==0.2.3

# ── Application source ─────────────────────────────────────────────────────────
COPY models.py /app/models.py
COPY server/ /app/server/

ENV PYTHONPATH="/app:/app/server"

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -sf http://localhost:7860/health || exit 1

CMD ["python", "server/app.py"]