# ─── Monsoon Flood Gate Control — OpenEnv Dockerfile ────────────────────────
# Build: docker build -t monsoon-floodgate-control:latest .
# Run:   docker run -p 8000:8000 monsoon-floodgate-control:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Optional: enable web UI
# ENV ENABLE_WEB_INTERFACE=true

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server/

# Copy spec file
COPY openenv.yaml .

# Set Python path so imports resolve
ENV PYTHONPATH=/app/server

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
