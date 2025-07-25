# Glitcher Docker Documentation

This document provides comprehensive instructions for using Glitcher with Docker, including development, production, and GPU-accelerated setups.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Image Variants](#image-variants)
5. [Using Docker Compose](#using-docker-compose)
6. [Makefile Commands](#makefile-commands)
7. [GPU Support](#gpu-support)
8. [Volume Management](#volume-management)
9. [Environment Configuration](#environment-configuration)
10. [Hugging Face Authentication](#hugging-face-authentication)
11. [Common Tasks](#common-tasks)
12. [Development Workflow](#development-workflow)
13. [Production Deployment](#production-deployment)
14. [Troubleshooting](#troubleshooting)

## Overview

The Glitcher Docker setup provides multiple optimized environments for different use cases:

- **Development**: Full environment with all tools and hot-reload capability
- **Production**: Minimal footprint for deployment
- **GPU**: Optimized for CUDA workloads with quantization support
- **CPU**: CPU-only version for environments without GPU
- **Web**: Web interface with Flask application

All images are built using multi-stage Docker builds for efficiency and security.

## Prerequisites

### System Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)
- 16GB+ RAM recommended
- 50GB+ free disk space

### GPU Requirements (Optional)

For GPU acceleration:
- NVIDIA GPU with CUDA 12.8+ support
- NVIDIA Container Toolkit installed
- PyTorch 2.7.1 with CUDA 12.8 compatibility
- 8GB+ VRAM recommended for 3B models
- 4GB+ VRAM minimum for 1B models

### Installing NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/binaryninja/glitcher.git
cd glitcher

# Create directories and check GPU
make setup

# Verify CUDA and PyTorch installation
make verify

# Setup Hugging Face authentication (for gated models like Llama)
# See HF-AUTH.md for detailed authentication guide
export HF_TOKEN=your_huggingface_token
# OR: make hf-setup TOKEN=your_huggingface_token

# Build and run a quick demo
make demo
```

### 2. Run Mining with GPU

```bash
# Build GPU image and run entropy-based mining
make mine

# Or with custom parameters
make mine MODEL=meta-llama/Llama-3.2-3B-Instruct
```

### 3. Start Development Environment

```bash
# Start interactive development container
make run-dev
```

### 4. Launch Web Interface

```bash
# Start web interface at http://localhost:5000
make run-web
```

## Image Variants

### Development Image (`glitcher:dev`)

**Purpose**: Full development environment with all tools and dependencies.

**Features**:
- All Python packages and extras installed
- Development tools (pytest, black, flake8, mypy)
- Jupyter Lab support
- Pre-commit hooks
- Hot-reload capability

**Build**: `make build-dev`

### Production Image (`glitcher:prod`)

**Purpose**: Minimal deployment image with only runtime dependencies.

**Features**:
- Core dependencies only
- Non-root user for security
- Optimized for size and security
- Health checks included

**Build**: `make build-prod`

### GPU Image (`glitcher:gpu`)

**Purpose**: Optimized for CUDA workloads with quantization support.

**Features**:
- CUDA 12.8.0 development environment (devel image)
- PyTorch 2.7.1 with CUDA acceleration
- BitsAndBytes for quantization
- Flash Attention and XFormers (compiled from source)
- Triton and Ninja for acceleration
- Optimized for high-performance inference

**Build**: `make build-gpu`

### CPU Image (`glitcher:cpu`)

**Purpose**: CPU-only version for environments without GPU.

**Features**:
- CPU-optimized PyTorch
- No GPU dependencies
- Suitable for basic tasks and testing

**Build**: `make build-cpu`

### Web Image (`glitcher:web`)

**Purpose**: Web interface with Flask application.

**Features**:
- Flask web framework
- Plotly for visualizations
- REST API endpoints
- Production WSGI server

**Build**: `make build-web`

## Using Docker Compose

Docker Compose provides orchestrated multi-container deployments with different profiles.

### Basic Services

```bash
# Start all basic services
docker-compose up -d

# Start specific service
docker-compose up glitcher-dev
docker-compose up glitcher-web
```

### Service Profiles

#### Development Profile

```bash
# Start development environment
docker-compose up glitcher-dev
```

#### Web Profile

```bash
# Start web interface
docker-compose up -d glitcher-web
# Access at http://localhost:5000
```

#### Jupyter Profile

```bash
# Start Jupyter Lab
docker-compose --profile jupyter up -d glitcher-jupyter
# Access at http://localhost:8888
```

#### Mining Profile

```bash
# Start mining services
docker-compose --profile mining up glitcher-miner
```

#### Analysis Profile

```bash
# Start classification and analysis
docker-compose --profile analysis up glitcher-classifier
```

#### Genetic Profile

```bash
# Start genetic algorithm optimization
docker-compose --profile genetic up glitcher-genetic
```

### Service Configuration

Services are configured through environment variables and volumes:

```yaml
# Example service configuration
glitcher-gpu:
  environment:
    - CUDA_VISIBLE_DEVICES=0
    - TOKENIZERS_PARALLELISM=false
  volumes:
    - ./models:/app/models
    - ./outputs:/app/outputs
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Makefile Commands

The Makefile provides convenient shortcuts for common Docker operations.

### Build Commands

```bash
make build-dev      # Build development image
make build-prod     # Build production image
make build-gpu      # Build GPU-optimized image
make build-cpu      # Build CPU-only image
make build-web      # Build web interface image
make build-all      # Build all images
```

### Run Commands

```bash
make run-dev        # Start development container with shell
make run-prod       # Start production container
make run-gpu        # Start GPU container with shell
make run-cpu        # Start CPU container with shell
make run-web        # Start web interface
make run-jupyter    # Start Jupyter Lab
```

### Mining Commands

```bash
make mine           # Run entropy-based mining
make mine-range     # Run range-based mining
make mine-unicode   # Run unicode-range mining
make mine-special   # Run special-token mining
make mine-config    # Run mining with configuration file
```

### Analysis Commands

```bash
make classify       # Run glitch token classification
make genetic        # Run genetic algorithm optimization
make genetic-gui    # Run genetic algorithm with GUI
make test-tokens    # Test specific token IDs
make validate       # Run validation suite
```

### Management Commands

```bash
make setup          # Create directories and check environment
make verify         # Verify CUDA and PyTorch installation
make logs           # Show container logs
make shell          # Open shell in development container
make stop           # Stop all containers
make clean          # Remove containers and images
make check-gpu      # Check GPU availability
```

### Development Commands

```bash
make lint           # Run code linting
make format         # Format code with black
make test-unit      # Run unit tests
make test-integration # Run integration tests
make test-all       # Run all tests
```

## GPU Support

### CUDA Configuration

The GPU images are built with CUDA 12.8.0 development environment and include:

- **PyTorch 2.7.1**: GPU-accelerated deep learning framework with CUDA 12.8 compatibility and latest optimizations
- **BitsAndBytes**: 4-bit and 8-bit quantization (requires CUDA 12.8+ for optimal performance)
- **Flash Attention**: Memory-efficient attention computation (compiled from source using nvcc)
- **XFormers**: Optimized transformer operations with CUDA 12.8 acceleration (compiled from source)
- **Triton**: GPU kernel compilation framework
- **Ninja**: Fast build system for compilation

### Memory Management

GPU memory is managed through environment variables:

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Configure memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Hugging Face authentication (required for gated models)
export HF_TOKEN=your_huggingface_token
```

### Quantization Support

The framework supports multiple quantization types:

- **int4**: 4-bit quantization (recommended for large models)
- **int8**: 8-bit quantization (balance of speed and quality)
- **float16**: Half precision (faster inference)
- **bfloat16**: Brain floating point (better numerical stability)

### Memory Requirements

| Model Size | int4 | int8 | float16 | bfloat16 |
|------------|------|------|---------|----------|
| 1B params | 1GB  | 2GB  | 4GB     | 4GB      |
| 3B params | 2GB  | 4GB  | 8GB     | 8GB      |
| 7B params | 4GB  | 8GB  | 16GB    | 16GB     |

### GPU Health Check

```bash
# Check GPU availability
make check-gpu

# Or manually
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

## Volume Management

### Directory Structure

The Docker setup uses three main volumes:

```
./models/     # Model files and cache
./outputs/    # Mining results and outputs
./data/       # Input data and configurations
```

### Volume Mapping

```bash
# Development (read-write)
-v $(PWD):/app                    # Source code
-v ./models:/app/models           # Model cache
-v ./outputs:/app/outputs         # Results
-v ./data:/app/data              # Input data

# Production (read-only models/data)
-v ./models:/app/models:ro        # Read-only models
-v ./outputs:/app/outputs         # Write outputs
-v ./data:/app/data:ro           # Read-only data
```

### Creating Volumes

```bash
# Create local directories
make setup-dirs

# Or manually
mkdir -p models outputs data
```

### Persistent Volumes

For production deployments, use Docker volumes:

```yaml
volumes:
  glitcher-models:
    driver: local
  glitcher-outputs:
    driver: local
  glitcher-data:
    driver: local
```

## Environment Configuration

### Environment Variables

Key environment variables for configuration:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0              # GPU device selection
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management

# Model Configuration
TOKENIZERS_PARALLELISM=false        # Disable tokenizer warnings
TRANSFORMERS_CACHE=/app/models      # Model cache directory
HF_HOME=/app/models                 # Hugging Face cache

# Application Configuration
GLITCHER_OUTPUTS_DIR=/app/outputs   # Output directory
GLITCHER_DATA_DIR=/app/data         # Data directory
GLITCHER_CONFIG=/app/config.json    # Configuration file

# Hugging Face Authentication
HF_TOKEN=your_huggingface_token     # HF API token for gated models
HF_HOME=/app/models                 # HF cache directory

# Web Interface
FLASK_ENV=production                # Flask environment
FLASK_APP=glitcher.web.app         # Flask application
```

### Configuration File

The `mining-config.json` file provides default settings:

```json
{
  "mining": {
    "default_model": "meta-llama/Llama-3.2-1B-Instruct",
    "modes": {
      "entropy": {
        "num_iterations": 50,
        "batch_size": 8,
        "k": 32,
        "asr_threshold": 0.5
      }
    }
  }
}
```

### Environment File

Create a `.env` file for persistent configuration:

```bash
# Create environment file
make setup-env

# Or manually create .env
cat > .env << EOF
MODELS_PATH=./models
OUTPUTS_PATH=./outputs
DATA_PATH=./data
CUDA_VISIBLE_DEVICES=0
HF_TOKEN=your_huggingface_token
EOF
```

## Hugging Face Authentication

> 📖 **Detailed Guide**: See [HF-AUTH.md](HF-AUTH.md) for comprehensive authentication setup and troubleshooting.

Many state-of-the-art models (like Llama, Mistral, etc.) are gated and require Hugging Face authentication.

### Getting a Hugging Face Token

1. **Create Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Generate Token**: Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Request Access**: Visit model pages (e.g., [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)) and request access

### Authentication Methods

#### Method 1: Environment Variable
```bash
# Set token in current session
export HF_TOKEN=hf_your_token_here

# Add to .env file for persistence
echo "HF_TOKEN=hf_your_token_here" >> .env
```

#### Method 2: Makefile Helper
```bash
# Interactive setup (saves to .env)
make hf-login

# Setup with token
make hf-setup TOKEN=hf_your_token_here

# Check authentication status
make hf-status
```

#### Method 3: Docker Authentication Script
```bash
# Run authentication helper
docker run --rm -it -v $(PWD):/app/host glitcher:dev \
  python /app/scripts/setup_hf_auth.py --login --save --env-file /app/host/.env
```

### Verifying Authentication

```bash
# Check authentication status
make hf-status

# Test with gated model
HF_TOKEN=your_token make mine MODEL=meta-llama/Llama-3.2-1B-Instruct
```

### Supported Gated Models

| Model Family | Example Model | Access Required |
|--------------|---------------|-----------------|
| **Llama 3.2** | `meta-llama/Llama-3.2-1B-Instruct` | ✅ Request access |
| **Llama 3.1** | `meta-llama/Meta-Llama-3.1-8B-Instruct` | ✅ Request access |
| **Mistral** | `mistralai/Mistral-7B-Instruct-v0.3` | ✅ Request access |
| **Public Models** | `microsoft/DialoGPT-medium` | ❌ No auth needed |

### Authentication Troubleshooting

#### Token Not Working
```bash
# Verify token validity
make hf-status

# Check token permissions at https://huggingface.co/settings/tokens
# Ensure token has "Read" permission for gated models
```

#### Access Denied
```bash
# Request access to specific models:
# 1. Visit model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
# 2. Click "Request access" button
# 3. Wait for approval (can take hours to days)
```

#### Environment Issues
```bash
# Check if token is properly set
docker run --rm -e HF_TOKEN=$HF_TOKEN glitcher:gpu \
  python -c "import os; print('Token:', os.getenv('HF_TOKEN', 'NOT SET')[:8] + '...')"

# Test token in container
docker run --rm -e HF_TOKEN=$HF_TOKEN glitcher:dev \
  python /app/scripts/setup_hf_auth.py --status
```

### Quick Authentication Reference

For detailed authentication setup, see **[HF-AUTH.md](HF-AUTH.md)**

## Common Tasks
</text>

<old_text line=518>
#### Basic Mining

```bash
# Entropy-based mining with default settings
make mine

# Custom model and parameters
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  glitcher:gpu \
  glitcher mine meta-llama/Llama-3.2-3B-Instruct \
    --num-iterations 100 \
    --batch-size 4 \
    --k 64 \
    --asr-threshold 0.8
```

### Mining Glitch Tokens

#### Basic Mining

```bash
# Entropy-based mining with default settings
make mine

# Custom model and parameters
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  glitcher:gpu \
  glitcher mine meta-llama/Llama-3.2-3B-Instruct \
    --num-iterations 100 \
    --batch-size 4 \
    --k 64 \
    --asr-threshold 0.8
```

#### Range-Based Mining

```bash
# Mine specific token ID ranges
# Range-based mining
HF_TOKEN=your_token make mine-range

# Custom range with authentication
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  -e HF_TOKEN=$HF_TOKEN \
  glitcher:gpu \
  glitcher mine meta-llama/Llama-3.2-1B-Instruct \
    --mode range \
    --range-start 128000 \
    --range-end 128256 \
    --sample-rate 1.0
```

#### Unicode Mining

```bash
# Mine Unicode character ranges
make mine-unicode

# With custom parameters
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  glitcher:gpu \
  glitcher mine meta-llama/Llama-3.2-1B-Instruct \
    --mode unicode \
    --sample-rate 0.1 \
    --max-tokens-per-range 100
```

### Classifying Glitch Tokens

```bash
# Basic classification
make classify

# Email extraction only
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  glitcher:gpu \
  glitch-classify meta-llama/Llama-3.2-1B-Instruct \
    --email-extraction-only \
    --max-tokens 500 \
    --token-file /app/outputs/glitch_tokens.json
```

### Genetic Algorithm Optimization

```bash
# Basic genetic optimization
make genetic

# With GUI (requires X11 forwarding)
make genetic-gui

# Custom parameters
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  glitcher:gpu \
  glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
    --base-text "Hello world" \
    --generations 100 \
    --population-size 50 \
    --target-token "specific_target"
```

### Testing Specific Tokens

```bash
# Test known glitch tokens
make test-tokens

# Custom token IDs
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  glitcher:gpu \
  glitcher test meta-llama/Llama-3.2-1B-Instruct \
    --token-ids 89472,127438,85069 \
    --enhanced \
    --num-attempts 5 \
    --asr-threshold 0.8
```

### Validation and Comparison

```bash
# Run validation suite
make validate

# Compare standard vs enhanced validation
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  glitcher:gpu \
  glitcher compare meta-llama/Llama-3.2-1B-Instruct \
    --token-ids 89472,127438,85069 \
    --num-attempts 5 \
    --asr-threshold 0.8
```

## Development Workflow

### Setting Up Development Environment

```bash
# 1. Start development container
make run-dev

# 2. Install additional dev dependencies (if needed)
pip install -e ".[all,dev]"

# 3. Set up pre-commit hooks
pre-commit install

# 4. Run tests to verify setup
pytest tests/
```

### Code Development Cycle

```bash
# 1. Edit code in host filesystem (changes reflected in container)
vim glitcher/cli.py

# 2. Run tests
make test-unit

# 3. Format code
make format

# 4. Run linting
make lint

# 5. Run integration tests
make test-integration
```

### Debugging

```bash
# Access running container
make shell

# Or specific container
make shell-gpu

# View logs
make logs

# Execute commands in container
make exec CMD='glitcher --help'
```

### Interactive Development

```bash
# Start Jupyter Lab
make run-jupyter
# Access at http://localhost:8888

# Start IPython in container
docker exec -it glitcher-dev ipython

# Run specific scripts
docker exec -it glitcher-dev python examples/test_mining.py
```

## Production Deployment

### Single Container Deployment

```bash
# Build production image
make build-prod

# Deploy with minimal resources
docker run -d \
  --name glitcher-prod \
  --restart unless-stopped \
  -v /data/models:/app/models:ro \
  -v /data/outputs:/app/outputs \
  glitcher:prod \
  glitcher mine meta-llama/Llama-3.2-1B-Instruct
```

### Multi-Container Deployment

```bash
# Production deployment with compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale specific services
docker-compose up -d --scale glitcher-miner=3
```

### Container Orchestration

For Kubernetes deployment, create appropriate manifests:

```yaml
# Example Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: glitcher-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: glitcher-gpu
  template:
    metadata:
      labels:
        app: glitcher-gpu
    spec:
      containers:
      - name: glitcher
        image: glitcher:gpu
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            memory: 8Gi
```

### Health Monitoring

```bash
# Check container health
docker ps
make stats

# View resource usage
make stats

# Check logs
make logs-prod
```

### Backup and Restore

```bash
# Create backup
make backup

# Restore from backup
make restore BACKUP=glitcher_backup_20231201_120000.tar.gz
```

## Troubleshooting

### Common Issues

#### GPU Not Available

**Problem**: CUDA not detected or GPU not accessible.

**Solutions**:
```bash
# Check NVIDIA driver (must support CUDA 12.8+)
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.8.0-devel-ubuntu22.04 nvidia-smi

# Verify PyTorch CUDA compatibility
docker run --rm --gpus all glitcher:gpu python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Restart Docker daemon
sudo systemctl restart docker

# Use CPU-only image as fallback
make run-cpu
```

#### Out of Memory Errors

**Problem**: CUDA out of memory or insufficient system RAM.

**Solutions**:
```bash
# Use smaller model
docker run ... glitcher mine meta-llama/Llama-3.2-1B-Instruct

# Reduce batch size
docker run ... glitcher mine MODEL --batch-size 2

# Use quantization
docker run ... glitcher mine MODEL --quant-type int4

# Clear GPU memory
docker restart glitcher-gpu
```

#### Permission Denied

**Problem**: Permission issues with volumes or files.

**Solutions**:
```bash
# Fix directory permissions
sudo chown -R $USER:$USER models outputs data

# Use correct user in container
docker run --user $(id -u):$(id -g) ...

# Check volume mounts
docker inspect glitcher-dev | grep Mounts -A 10
```

#### Container Won't Start

**Problem**: Container fails to start or exits immediately.

**Solutions**:
```bash
# Check logs for errors
docker logs glitcher-dev

# Verify image build
docker run --rm -it glitcher:dev bash

# Check resource limits
docker stats

# Rebuild image
make clean build-dev
```

#### Model Download Issues

**Problem**: Model fails to download or load.

**Solutions**:
```bash
# Check internet connectivity
docker run --rm glitcher:dev ping huggingface.co

# Check HF authentication
make hf-status

# Pre-download models with authentication
HF_TOKEN=your_token docker run --rm -v ./models:/app/models -e HF_TOKEN=$HF_TOKEN glitcher:dev \
  python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"

# Use local model path
docker run -v /local/models:/app/models ... glitcher mine /app/models/llama-3.2-1b
```

#### Hugging Face Authentication Issues

**Problem**: Cannot access gated models like Llama.

**Solutions**:
```bash
# Check authentication status
make hf-status

# Verify token is set
echo $HF_TOKEN

# Test token validity
make hf-setup TOKEN=$HF_TOKEN

# Request model access
# Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# Click "Request access" and wait for approval
```

#### Web Interface Issues

**Problem**: Web interface not accessible or not working.

**Solutions**:
```bash
# Check if service is running
docker ps | grep glitcher-web

# Check port mapping
docker port glitcher-web

# View web logs
make logs-web

# Test health endpoint
curl http://localhost:5000/health

# Start with HF authentication
HF_TOKEN=your_token make run-web
```

### Performance Optimization

#### GPU Performance

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Optimize memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use mixed precision
# (automatically enabled in GPU images)
```

#### CPU Performance

```bash
# Set CPU affinity
docker run --cpuset-cpus="0-3" ...

# Limit memory usage
docker run --memory="8g" ...

# Use optimized BLAS libraries
# (included in base images)
```

#### Storage Performance

```bash
# Use SSD for model storage
# Mount SSD to /app/models

# Use tmpfs for temporary files
docker run --tmpfs /tmp:rw,noexec,nosuid,size=1g ...

# Optimize Docker storage driver
# Use overlay2 or fuse-overlayfs
```

### Debug Mode

Enable debug logging and verbose output:

```bash
# Set debug environment
export GLITCHER_DEBUG=1
export TRANSFORMERS_VERBOSITY=debug

# Run with debug flags
docker run -e GLITCHER_DEBUG=1 -e TRANSFORMERS_VERBOSITY=debug ...
```

### Getting Help

```bash
# Show help for any command
make help

# Container-specific help
docker run --rm glitcher:gpu glitcher --help
docker run --rm glitcher:gpu glitcher mine --help

# Check documentation
docker run --rm glitcher:dev cat /app/README.md
```

For additional support:
- Check the [main documentation](README.md)
- Review [CLAUDE.md](CLAUDE.md) for detailed usage
- Open issues on the [GitHub repository](https://github.com/binaryninja/glitcher/issues)

## Advanced Usage

### Custom Images

Build custom images with additional dependencies:

```dockerfile
# Dockerfile.custom
FROM glitcher:gpu
USER root
RUN pip install custom-package
USER glitcher
```

```bash
# Build custom image
docker build -f Dockerfile.custom -t glitcher:custom .
```

### Multi-GPU Setup

For multi-GPU deployments:

```bash
# Use multiple GPUs
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0,1 ...

# GPU-specific containers
docker run --gpus '"device=0"' -e CUDA_VISIBLE_DEVICES=0 --name gpu0 ...
docker run --gpus '"device=1"' -e CUDA_VISIBLE_DEVICES=1 --name gpu1 ...
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Docker Build and Test
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build and test
      run: |
        make build-cpu
        make test-unit
  build-gpu:
    runs-on: [self-hosted, gpu, cuda-12.8]
    steps:
    - uses: actions/checkout@v3
    - name: Build and test GPU image (with CUDA devel tools)
      run: |
        make build-gpu
        make test-integration
```

This completes the comprehensive Docker documentation for the Glitcher project.