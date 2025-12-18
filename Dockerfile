# Multi-stage Dockerfile for Voice MCP Agent
# Self-contained image with VibeVoice baked in

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.11

# =============================================================================
# Stage 1: Base image with CUDA + system dependencies
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    build-essential \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS (required for Playwright MCP)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Install Python dependencies
# =============================================================================
FROM base AS python-deps

WORKDIR /build

# Copy requirements first for layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install main requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install additional audio/ML dependencies needed by VibeVoice
RUN pip install --no-cache-dir \
    numba \
    llvmlite \
    diffusers \
    ml-collections \
    absl-py \
    av

# =============================================================================
# Stage 3: Install VibeVoice (baked into image)
# =============================================================================
FROM python-deps AS vibevoice

# Copy VibeVoice source code
COPY VibeVoice/ /opt/VibeVoice/

# Install VibeVoice in editable mode (dependencies already installed)
WORKDIR /opt/VibeVoice
RUN pip install --no-cache-dir -e . --no-deps

# =============================================================================
# Stage 4: Final application image
# =============================================================================
FROM vibevoice AS app

WORKDIR /app

# Copy application code
COPY web.py mcp_agent.py mcp_agent_config.json ./
COPY voice/ ./voice/

# Set Python path to include VibeVoice
ENV PYTHONPATH="/opt/VibeVoice:${PYTHONPATH}"

# Set voice presets path (baked into image)
ENV VOICE_PRESETS_PATH="/opt/VibeVoice/demo/voices/streaming_model"

# Default environment variables (can be overridden at runtime)
ENV LLM_BASE_URL="http://vllm:8000/v1"
ENV LLM_API_KEY="not-needed"
ENV LLM_MODEL="devstral-2-24b"
ENV ASR_DEVICE="cuda:0"
ENV TTS_DEVICE="cuda:1"
ENV VOICE_ENABLED="true"

# Create cache directory for HuggingFace models
RUN mkdir -p /root/.cache/huggingface

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Default command
CMD ["python", "web.py", "--host", "0.0.0.0", "--port", "3000"]
