# Use official Python 3.13 slim image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy only requirements metadata first to leverage Docker cache
COPY pyproject.toml poetry.lock* /app/

# Install system dependencies required for OpenEnv and some packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install OpenEnv CLI and dependencies
RUN pip install openenv-core[cli]==0.2.3

# Copy the whole project into the container
COPY . /app

# Expose the port used by your app (example: 7860 for Gradio)
EXPOSE 7860

# Default command to run your app
CMD ["python", "server/app.py"]