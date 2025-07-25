# Glitcher Docker Makefile
# Provides convenient commands for building and running Docker containers

.PHONY: help build build-dev build-prod build-gpu build-cpu build-web clean clean-all
.PHONY: run run-dev run-prod run-gpu run-cpu run-web run-jupyter
.PHONY: mine classify genetic test validate
.PHONY: logs shell exec stop restart
.PHONY: setup-dirs setup-env check-gpu install-dev
.PHONY: lint test-unit test-integration
.PHONY: backup restore

# Default target
.DEFAULT_GOAL := help

# Configuration variables
DOCKER_REGISTRY ?= ghcr.io/binaryninja
IMAGE_NAME ?= glitcher
VERSION ?= latest
BUILD_TARGET ?= production

# Directories
MODELS_DIR ?= ./models
OUTPUTS_DIR ?= ./outputs
DATA_DIR ?= ./data

# GPU configuration
GPU_DEVICE ?= 0
CUDA_VISIBLE_DEVICES ?= $(GPU_DEVICE)

# Model configuration
DEFAULT_MODEL ?= meta-llama/Llama-3.2-1B-Instruct

# Hugging Face authentication
HF_TOKEN ?= $(shell echo $$HF_TOKEN)

help: ## Show this help message
	@echo "Glitcher Docker Management"
	@echo "=========================="
	@echo ""
	@echo "Build Commands:"
	@echo "  build-dev     Build development image with all tools"
	@echo "  build-prod    Build production image (minimal)"
	@echo "  build-gpu     Build GPU-optimized image"
	@echo "  build-cpu     Build CPU-only image"
	@echo "  build-web     Build web interface image"
	@echo "  build-all     Build all images"
	@echo ""
	@echo "Environment Variables:"
	@echo "  HF_TOKEN      Hugging Face API token for gated models"
	@echo ""
	@echo "Run Commands:"
	@echo "  run-dev       Start development container with shell"
	@echo "  run-prod      Start production container"
	@echo "  run-gpu       Start GPU-optimized container"
	@echo "  run-cpu       Start CPU-only container"
	@echo "  run-web       Start web interface (http://localhost:5000)"
	@echo "  run-jupyter   Start Jupyter lab (http://localhost:8888)"
	@echo ""
	@echo "Mining Commands:"
	@echo "  mine          Run entropy-based mining"
	@echo "  mine-range    Run range-based mining"
	@echo "  mine-unicode  Run unicode-range mining"
	@echo "  mine-special  Run special-token mining"
	@echo "  deep-mine     Run comprehensive deep mining pipeline"
	@echo "  deep-quick    Run quick deep mining test"
	@echo ""
	@echo "Analysis Commands:"
	@echo "  classify      Run glitch token classification"
	@echo "  genetic       Run genetic algorithm optimization"
	@echo "  test-tokens   Test specific token IDs"
	@echo "  validate      Run validation suite"
	@echo ""
	@echo "Authentication Commands:"
	@echo "  hf-status     Check Hugging Face authentication status"
	@echo "  hf-login      Interactive Hugging Face login"
	@echo "  hf-setup      Setup HF authentication with token"
	@echo ""
	@echo "Management Commands:"
	@echo "  setup         Create directories and check environment"
	@echo "  verify        Verify CUDA and PyTorch installation"
	@echo "  logs          Show container logs"
	@echo "  shell         Open shell in running container"
	@echo "  stop          Stop all containers"
	@echo "  clean         Remove containers and images"
	@echo "  check-gpu     Check GPU availability"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Build targets
build-dev: ## Build development image
	docker build --target development -t $(IMAGE_NAME):dev .

build-prod: ## Build production image
	docker build --target production -t $(IMAGE_NAME):prod .

build-gpu: ## Build GPU-optimized image
	docker build --target gpu -t $(IMAGE_NAME):gpu .

build-cpu: ## Build CPU-only image
	docker build --target cpu -t $(IMAGE_NAME):cpu .

build-web: ## Build web interface image
	docker build --target web -t $(IMAGE_NAME):web .

build-all: build-dev build-prod build-gpu build-cpu build-web ## Build all images

build: build-prod ## Build default (production) image

# Setup and environment
setup: setup-dirs check-gpu ## Setup environment and directories
	@echo "Environment setup complete"

setup-dirs: ## Create necessary directories
	@echo "Creating directories..."
	@mkdir -p $(MODELS_DIR) $(OUTPUTS_DIR) $(DATA_DIR)
	@echo "Directories created: $(MODELS_DIR), $(OUTPUTS_DIR), $(DATA_DIR)"

check-gpu: ## Check GPU availability
	@echo "Checking GPU availability..."
	@docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi || \
		echo "GPU not available - use CPU-only targets"

verify: build-gpu setup-dirs ## Verify CUDA and PyTorch installation
	@echo "Running CUDA and PyTorch verification..."
	docker run --rm \
		--gpus all \
		-v $(PWD)/scripts:/app/scripts \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(IMAGE_NAME):gpu \
		python /app/scripts/verify_cuda_pytorch.py

verify-cpu: build-cpu setup-dirs ## Verify CPU-only installation
	@echo "Running CPU-only verification..."
	docker run --rm \
		-v $(PWD)/scripts:/app/scripts \
		$(IMAGE_NAME):cpu \
		python /app/scripts/verify_cuda_pytorch.py

setup-env: ## Create example environment file
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		echo "MODELS_PATH=$(MODELS_DIR)" > .env; \
		echo "OUTPUTS_PATH=$(OUTPUTS_DIR)" >> .env; \
		echo "DATA_PATH=$(DATA_DIR)" >> .env; \
		echo "CUDA_VISIBLE_DEVICES=$(GPU_DEVICE)" >> .env; \
		echo "HF_TOKEN=" >> .env; \
		echo "Created .env file"; \
		echo "Don't forget to add your Hugging Face token to HF_TOKEN in .env"; \
	else \
		echo ".env file already exists"; \
	fi

# Run targets
run-dev: build-dev setup-dirs ## Start development container
	docker run -it --rm \
		--gpus all \
		-v $(PWD):/app \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		--name glitcher-dev \
		$(IMAGE_NAME):dev

run-prod: build-prod setup-dirs ## Start production container
	docker run -it --rm \
		-v $(MODELS_DIR):/app/models:ro \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data:ro \
		-e HF_TOKEN=$(HF_TOKEN) \
		--name glitcher-prod \
		$(IMAGE_NAME):prod

run-gpu: build-gpu setup-dirs ## Start GPU container
	docker run -it --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		--name glitcher-gpu \
		$(IMAGE_NAME):gpu bash

run-cpu: build-cpu setup-dirs ## Start CPU container
	docker run -it --rm \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data \
		-e HF_TOKEN=$(HF_TOKEN) \
		--name glitcher-cpu \
		$(IMAGE_NAME):cpu bash

run-web: build-web setup-dirs ## Start web interface
	docker run -d \
		-p 5000:5000 \
		-v $(MODELS_DIR):/app/models:ro \
		-v $(OUTPUTS_DIR):/app/outputs:ro \
		-v $(DATA_DIR):/app/data:ro \
		-e HF_TOKEN=$(HF_TOKEN) \
		--name glitcher-web \
		$(IMAGE_NAME):web
	@echo "Web interface available at http://localhost:5000"

run-jupyter: build-dev setup-dirs ## Start Jupyter lab
	docker run -d \
		--gpus all \
		-p 8888:8888 \
		-v $(PWD):/app \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		--name glitcher-jupyter \
		$(IMAGE_NAME):dev \
		jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
	@echo "Jupyter lab available at http://localhost:8888"

# Mining commands
mine: build-gpu setup-dirs ## Run entropy-based mining
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitcher mine $(DEFAULT_MODEL) --num-iterations 50 --batch-size 8 --k 32

mine-range: build-gpu setup-dirs ## Run range-based mining
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitcher mine $(DEFAULT_MODEL) --mode range --range-start 0 --range-end 1000 --sample-rate 0.1

mine-unicode: build-gpu setup-dirs ## Run unicode-range mining
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitcher mine $(DEFAULT_MODEL) --mode unicode --sample-rate 0.05 --max-tokens-per-range 50

mine-special: build-gpu setup-dirs ## Run special-token mining
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitcher mine $(DEFAULT_MODEL) --mode special --sample-rate 0.2 --max-tokens-per-range 100

mine-config: build-gpu setup-dirs ## Run mining with configuration file
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(PWD)/mining-config.json:/app/config.json:ro \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e GLITCHER_CONFIG=/app/config.json \
		$(IMAGE_NAME):gpu \
		glitcher mine $(DEFAULT_MODEL)

# Analysis commands
classify: build-gpu setup-dirs ## Run glitch token classification
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitch-classify $(DEFAULT_MODEL) --max-tokens 500 --token-file /app/outputs/glitch_tokens.json

genetic: build-gpu setup-dirs ## Run genetic algorithm optimization
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitcher genetic $(DEFAULT_MODEL) --base-text "The quick brown" --generations 50 --population-size 30

genetic-gui: build-gpu setup-dirs ## Run genetic algorithm with GUI
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-v $(DATA_DIR):/app/data \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-e DISPLAY=$(DISPLAY) \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		--network host \
		$(IMAGE_NAME):gpu \
		glitcher genetic $(DEFAULT_MODEL) --gui --base-text "The quick brown" --generations 50

test-tokens: build-gpu setup-dirs ## Test specific token IDs
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitcher test $(DEFAULT_MODEL) --token-ids 89472,127438,85069 --enhanced --num-attempts 3

validate: build-gpu setup-dirs ## Run validation suite
	docker run --rm \
		--gpus all \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):gpu \
		glitcher validate $(DEFAULT_MODEL) --output-dir /app/outputs/validation_results

# Docker Compose shortcuts
compose-up: ## Start all services with docker-compose
	docker-compose up -d

compose-down: ## Stop all services
	docker-compose down

compose-dev: ## Start development environment
	docker-compose up glitcher-dev

compose-web: ## Start web interface
	docker-compose up -d glitcher-web
	@echo "Web interface available at http://localhost:5000"

compose-jupyter: ## Start Jupyter service
	docker-compose --profile jupyter up -d glitcher-jupyter
	@echo "Jupyter lab available at http://localhost:8888"

compose-mining: ## Start mining services
	docker-compose --profile mining up glitcher-miner

compose-analysis: ## Start analysis services
	docker-compose --profile analysis up glitcher-classifier

# Management commands
logs: ## Show container logs
	@echo "Available containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
	@echo ""
	@echo "Use 'make logs-<container>' to see specific logs, e.g.:"
	@echo "  make logs-dev, make logs-web, make logs-gpu"

logs-%: ## Show logs for specific container
	docker logs -f glitcher-$*

shell: ## Open shell in development container
	docker exec -it glitcher-dev bash

shell-%: ## Open shell in specific container
	docker exec -it glitcher-$* bash

exec: ## Execute command in development container
	@echo "Usage: make exec CMD='your command'"
	@if [ "$(CMD)" = "" ]; then \
		echo "Example: make exec CMD='glitcher --help'"; \
	else \
		docker exec glitcher-dev $(CMD); \
	fi

stop: ## Stop all glitcher containers
	@echo "Stopping all glitcher containers..."
	@docker ps --filter "name=glitcher-*" -q | xargs -r docker stop
	@echo "All containers stopped"

stop-%: ## Stop specific container
	docker stop glitcher-$*

restart: stop compose-up ## Restart all services

restart-%: ## Restart specific service
	docker-compose restart glitcher-$*

# Cleanup commands
clean: ## Remove containers and images
	@echo "Stopping containers..."
	@docker ps --filter "name=glitcher-*" -q | xargs -r docker stop
	@echo "Removing containers..."
	@docker ps -a --filter "name=glitcher-*" -q | xargs -r docker rm
	@echo "Removing images..."
	@docker images $(IMAGE_NAME) -q | xargs -r docker rmi
	@echo "Cleanup complete"

clean-all: clean ## Remove everything including volumes
	@echo "Removing volumes..."
	@docker volume ls --filter "name=glitcher-*" -q | xargs -r docker volume rm
	@echo "Complete cleanup finished"

prune: ## Remove unused Docker resources
	docker system prune -af
	docker volume prune -f

# Development commands
install-dev: build-dev ## Install development dependencies
	docker run --rm -v $(PWD):/app $(IMAGE_NAME):dev pip install -e ".[all,dev]"

lint: build-dev ## Run code linting
	docker run --rm -v $(PWD):/app $(IMAGE_NAME):dev bash -c "black --check . && flake8 ."

format: build-dev ## Format code
	docker run --rm -v $(PWD):/app $(IMAGE_NAME):dev black .

test-unit: build-dev ## Run unit tests
	docker run --rm -v $(PWD):/app $(IMAGE_NAME):dev pytest tests/unit/

test-integration: build-gpu ## Run integration tests
	docker run --rm \
		--gpus all \
		-v $(PWD):/app \
		-v $(MODELS_DIR):/app/models \
		-v $(OUTPUTS_DIR):/app/outputs \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(IMAGE_NAME):gpu \
		pytest tests/integration/

test-all: test-unit test-integration ## Run all tests

# Backup and restore
backup: ## Backup outputs and data
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	echo "Creating backup: glitcher_backup_$$timestamp.tar.gz"; \
	tar -czf glitcher_backup_$$timestamp.tar.gz $(OUTPUTS_DIR) $(DATA_DIR) *.json; \
	echo "Backup created: glitcher_backup_$$timestamp.tar.gz"

restore: ## Restore from backup (Usage: make restore BACKUP=filename.tar.gz)
	@if [ "$(BACKUP)" = "" ]; then \
		echo "Usage: make restore BACKUP=glitcher_backup_YYYYMMDD_HHMMSS.tar.gz"; \
		ls -la glitcher_backup_*.tar.gz 2>/dev/null || echo "No backup files found"; \
	else \
		echo "Restoring from $(BACKUP)..."; \
		tar -xzf $(BACKUP); \
		echo "Restore complete"; \
	fi

# Registry operations
tag: ## Tag image for registry
	docker tag $(IMAGE_NAME):prod $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker tag $(IMAGE_NAME):gpu $(DOCKER_REGISTRY)/$(IMAGE_NAME):gpu-$(VERSION)

push: tag ## Push images to registry
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):gpu-$(VERSION)

pull: ## Pull images from registry
	docker pull $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker pull $(DOCKER_REGISTRY)/$(IMAGE_NAME):gpu-$(VERSION)

# Documentation
docs: build-dev ## Generate documentation
	docker run --rm -v $(PWD):/app $(IMAGE_NAME):dev bash -c "cd docs && make html"

# Monitoring
stats: ## Show container resource usage
	docker stats $(shell docker ps --filter "name=glitcher-*" --format "{{.Names}}")

ps: ## Show running containers
	docker ps --filter "name=glitcher-*"

images: ## Show glitcher images
	docker images $(IMAGE_NAME)

# Configuration validation
validate-config: ## Validate mining configuration
	@if [ -f mining-config.json ]; then \
		echo "Validating mining-config.json..."; \
		python -c "import json; json.load(open('mining-config.json')); print('✓ Valid JSON')"; \
	else \
		echo "mining-config.json not found"; \
	fi

# Quick start targets
quickstart: setup build-gpu mine ## Quick start: setup, build, and run mining
	@echo "Quick start complete! Check outputs/ directory for results."

demo: setup build-gpu ## Run a quick demonstration
	@echo "Running glitcher demonstration..."
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Using public model for demo..."; \
		docker run --rm \
			--gpus all \
			-v $(MODELS_DIR):/app/models \
			-v $(OUTPUTS_DIR):/app/outputs \
			-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
			$(IMAGE_NAME):gpu \
			glitcher mine microsoft/DialoGPT-medium --num-iterations 3 --batch-size 2; \
	else \
		echo "Using Llama model with HF_TOKEN..."; \
		docker run --rm \
			--gpus all \
			-v $(MODELS_DIR):/app/models \
			-v $(OUTPUTS_DIR):/app/outputs \
			-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
			-e HF_TOKEN=$(HF_TOKEN) \
			$(IMAGE_NAME):gpu \
			glitcher mine $(DEFAULT_MODEL) --num-iterations 3 --batch-size 2; \
	fi
	@echo "Demo complete! Check outputs/ directory for results."

# Authentication commands
hf-status: build-dev ## Check Hugging Face authentication status
	@echo "Checking Hugging Face authentication status..."
	docker run --rm \
		-v $(PWD)/scripts:/app/scripts \
		-e HF_TOKEN=$(HF_TOKEN) \
		$(IMAGE_NAME):dev \
		python /app/scripts/setup_hf_auth.py --status

hf-login: build-dev ## Interactive Hugging Face login
	@echo "Starting interactive Hugging Face login..."
	docker run --rm -it \
		-v $(PWD)/scripts:/app/scripts \
		-v $(PWD):/app/host \
		$(IMAGE_NAME):dev \
		python /app/scripts/setup_hf_auth.py --login --save --env-file /app/host/.env

hf-setup: build-dev ## Setup HF authentication with token (Usage: make hf-setup TOKEN=your_token)
	@if [ "$(TOKEN)" = "" ]; then \
		echo "Usage: make hf-setup TOKEN=your_huggingface_token"; \
		echo "Get a token at: https://huggingface.co/settings/tokens"; \
	else \
		echo "Setting up Hugging Face authentication..."; \
		docker run --rm \
			-v $(PWD)/scripts:/app/scripts \
			-v $(PWD):/app/host \
			$(IMAGE_NAME):dev \
			python /app/scripts/setup_hf_auth.py --token $(TOKEN) --save --env-file /app/host/.env; \
	fi

# Deep mining commands
deep-mine: build-gpu setup-dirs ## Run comprehensive deep mining pipeline
	@echo "Starting comprehensive deep mining pipeline..."
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Warning: No HF_TOKEN set. Using public model fallback."; \
		MODEL=microsoft/DialoGPT-medium; \
	else \
		MODEL=$(DEFAULT_MODEL); \
	fi; \
	HF_TOKEN=$(HF_TOKEN) \
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(PWD)/scripts/deep_mining_pipeline.sh \
		--model $$MODEL \
		--output $(OUTPUTS_DIR)/deep_mining \
		--gpu $(GPU_DEVICE)

deep-quick: build-gpu setup-dirs ## Run quick deep mining test
	@echo "Starting quick deep mining test..."
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Using public model for quick test."; \
		MODEL=microsoft/DialoGPT-medium; \
	else \
		MODEL=$(DEFAULT_MODEL); \
	fi; \
	HF_TOKEN=$(HF_TOKEN) \
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(PWD)/scripts/deep_mining_pipeline.sh \
		--model $$MODEL \
		--output $(OUTPUTS_DIR)/deep_mining \
		--gpu $(GPU_DEVICE) \
		--quick

deep-stage: build-gpu setup-dirs ## Run specific deep mining stage (Usage: make deep-stage STAGE=entropy)
	@if [ "$(STAGE)" = "" ]; then \
		echo "Usage: make deep-stage STAGE=<entropy|range|unicode|special|analysis>"; \
		echo "Available stages:"; \
		echo "  entropy  - Enhanced entropy-based mining"; \
		echo "  range    - Systematic token ID range exploration"; \
		echo "  unicode  - Unicode character range mining"; \
		echo "  special  - Special token and vocabulary edge mining"; \
		echo "  analysis - Results analysis and reporting"; \
	else \
		echo "Running deep mining stage: $(STAGE)"; \
		if [ -z "$(HF_TOKEN)" ]; then \
			echo "Warning: No HF_TOKEN set. Using public model fallback."; \
			MODEL=microsoft/DialoGPT-medium; \
		else \
			MODEL=$(DEFAULT_MODEL); \
		fi; \
		HF_TOKEN=$(HF_TOKEN) \
		CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(PWD)/scripts/deep_mining_pipeline.sh \
			--model $$MODEL \
			--output $(OUTPUTS_DIR)/deep_mining \
			--gpu $(GPU_DEVICE) \
			--stage $(STAGE); \
	fi
