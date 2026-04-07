# ── ATC TRACON RL Environment — Dockerfile ────────────────────────────────────
# Builds the FastAPI server that exposes the RL environment over HTTP.
# The image is intentionally kept slim; all heavy ML inference runs outside
# the container via inference.py talking to the server over the network.

FROM python:3.11-slim

# Prevent .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# Install server requirements before copying source so Docker can cache this layer
COPY my_env/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenEnv CLI
RUN pip install --no-cache-dir openenv-core==0.2.3

# ── Application source ─────────────────────────────────────────────────────────
# Copy the package models (shared between client and server)
COPY my_env/models.py /app/models.py

# Copy server code
COPY my_env/server/ /app/server/

# Make server modules importable from /app/server
ENV PYTHONPATH="/app:/app/server"

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 7860

# Healthcheck so orchestrators know when the server is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -sf http://localhost:7860/health || exit 1

CMD ["python", "server/app.py"]