# Multi-stage Dockerfile for glitcher with GPU support
# Supports both development and production scenarios

# Build arguments
ARG HF_TOKEN=""

# Base stage with CUDA and Python
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Dependencies stage - install Python packages
FROM base AS dependencies

# Copy setup.py and requirements first for better layer caching
COPY setup.py README.md ./

# Install core dependencies
RUN pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the package with core dependencies
RUN pip install -e .

# Development stage - includes dev tools and all extras
FROM dependencies AS development

# Install all development dependencies and extras
RUN pip install -e ".[all,dev]"

# Install additional development tools
RUN pip install \
    ipython \
    jupyter \
    pre-commit \
    mypy

# Copy the entire project
COPY . .

# Set up git hooks for development
RUN git init . || true && \
    pre-commit install || true

# Production stage - minimal for deployment
FROM dependencies AS production

# Copy only necessary files
COPY glitcher/ ./glitcher/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY poc/ ./poc/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash glitcher
RUN chown -R glitcher:glitcher /app
USER glitcher

# Set up environment variables for user packages
ENV PATH="/home/glitcher/.local/bin:$PATH"
ENV PYTHONPATH="/home/glitcher/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Reinstall package as glitcher user to fix entry points
RUN pip install --user -e .

# Verify installation
RUN python -c "import glitcher; print('Glitcher installed successfully')"

# GPU support stage - optimized for CUDA workloads
FROM production AS gpu

# Switch back to root to install GPU-optimized packages
USER root

# Install optimized GPU packages (now with devel image for compilation)
RUN pip install \
    bitsandbytes \
    flash-attn \
    xformers

# Install additional GPU acceleration libraries
RUN pip install \
    triton \
    ninja

# Verify GPU support
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')" || true
RUN python -c "import bitsandbytes; print('BitsAndBytes installed')" || true

# Switch back to non-root user
USER glitcher

# CPU-only stage - for environments without GPU
FROM production AS cpu

# Install CPU-only PyTorch
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Remove GPU-specific packages that won't work on CPU
RUN pip uninstall -y bitsandbytes || true

# Web interface stage - includes web dependencies
FROM production AS web

USER root
RUN pip install -e ".[web]"

# Expose port for web interface
EXPOSE 5000

USER glitcher

# Set default command
CMD ["glitcher", "--help"]

# Labels for metadata
LABEL maintainer="Binary Ninja" \
      version="0.1.0" \
      description="Glitcher - A tool for mining and testing glitch tokens in language models" \
      gpu_support="true" \
      python_version="3.10" \
      cuda_version="12.8.0" \
      pytorch_version="2.7.1"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import glitcher; print('OK')" || exit 1

# Environment variables for runtime configuration
ENV GLITCHER_HOME=/app \
    PYTHONPATH=/app:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES="" \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/models \
    HF_TOKEN=""

# Volume for persistent data
VOLUME ["/app/outputs", "/app/models", "/app/data"]
