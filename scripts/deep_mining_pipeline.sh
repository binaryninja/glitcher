#!/bin/bash
# Advanced Multi-Stage Deep Mining Pipeline for Glitcher
# Performs comprehensive glitch token discovery using multiple mining strategies

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/outputs/deep_mining"
MODELS_DIR="$PROJECT_ROOT/models"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/deep_mining_${TIMESTAMP}.log"

# Default parameters
MODEL="meta-llama/Llama-3.2-1B-Instruct"
DOCKER_IMAGE="glitcher:gpu"
GPU_DEVICE="0"
RESUME_MODE=false
STAGE_FILTER=""
QUICK_MODE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC}  $timestamp - $message" | tee -a "$LOG_FILE" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  $timestamp - $message" | tee -a "$LOG_FILE" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $timestamp - $message" | tee -a "$LOG_FILE" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} $timestamp - $message" | tee -a "$LOG_FILE" ;;
        "STAGE") echo -e "${PURPLE}[STAGE]${NC} $timestamp - $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Help function
show_help() {
    cat << EOF
Advanced Deep Mining Pipeline for Glitcher

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -m, --model MODEL           Model to mine (default: meta-llama/Llama-3.2-1B-Instruct)
    -o, --output DIR           Output directory (default: ./outputs/deep_mining)
    -g, --gpu DEVICE           GPU device number (default: 0)
    -r, --resume               Resume from previous run
    -s, --stage STAGE          Run only specific stage (entropy,range,unicode,special,analysis)
    -q, --quick                Quick mode - reduced parameters for testing
    -h, --help                 Show this help

ENVIRONMENT VARIABLES:
    HF_TOKEN                   Hugging Face API token (required for gated models)
    CUDA_VISIBLE_DEVICES       GPU device selection

EXAMPLES:
    # Full deep mining pipeline
    HF_TOKEN=hf_your_token $0 --model meta-llama/Llama-3.2-1B-Instruct

    # Quick test run
    HF_TOKEN=hf_your_token $0 --quick --model microsoft/DialoGPT-medium

    # Resume previous run
    $0 --resume

    # Run only Unicode stage
    HF_TOKEN=hf_your_token $0 --stage unicode

STAGES:
    1. entropy  - Enhanced entropy-based mining (baseline)
    2. range    - Systematic token ID range exploration
    3. unicode  - Unicode character range mining
    4. special  - Special token and vocabulary edge mining
    5. analysis - Results analysis and reporting

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model)
                MODEL="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -g|--gpu)
                GPU_DEVICE="$2"
                shift 2
                ;;
            -r|--resume)
                RESUME_MODE=true
                shift
                ;;
            -s|--stage)
                STAGE_FILTER="$2"
                shift 2
                ;;
            -q|--quick)
                QUICK_MODE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Setup environment
setup_environment() {
    log "INFO" "Setting up deep mining environment..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Check HF token for gated models
    if [[ "$MODEL" == *"meta-llama"* ]] || [[ "$MODEL" == *"mistral"* ]]; then
        if [[ -z "${HF_TOKEN:-}" ]]; then
            log "ERROR" "HF_TOKEN required for gated model: $MODEL"
            log "INFO" "Get token at: https://huggingface.co/settings/tokens"
            exit 1
        fi
        log "INFO" "Using HF_TOKEN for gated model access"
    fi

    # Check Docker image
    if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        log "ERROR" "Docker image $DOCKER_IMAGE not found. Run 'make build-gpu' first."
        exit 1
    fi

    # Check GPU availability
    if ! docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        log "WARN" "GPU not available, falling back to CPU"
        DOCKER_IMAGE="glitcher:cpu"
    fi

    log "INFO" "Environment setup complete"
    log "INFO" "Model: $MODEL"
    log "INFO" "Output: $OUTPUT_DIR"
    log "INFO" "Docker: $DOCKER_IMAGE"
    log "INFO" "GPU: $GPU_DEVICE"
}

# Run mining stage
run_mining_stage() {
    local stage=$1
    local config_name=$2
    local docker_args=("$@")
    docker_args=("${docker_args[@]:2}")  # Remove first two elements

    log "STAGE" "Starting $stage mining ($config_name)"

    local output_file="$OUTPUT_DIR/${stage}_${config_name}_${TIMESTAMP}.json"
    local progress_file="$OUTPUT_DIR/${stage}_${config_name}_progress.json"

    # Add resume flag if enabled and progress file exists
    local resume_args=()
    if [[ "$RESUME_MODE" == "true" ]] && [[ -f "$progress_file" ]]; then
        resume_args=(--resume --progress-file "$progress_file")
        log "INFO" "Resuming from: $progress_file"
    fi

    # Build Docker command
    local docker_cmd=(
        docker run --rm
        --gpus all
        -v "$MODELS_DIR:/app/models"
        -v "$OUTPUT_DIR:/app/outputs"
        -e "CUDA_VISIBLE_DEVICES=$GPU_DEVICE"
        -e "HF_TOKEN=${HF_TOKEN:-}"
        "$DOCKER_IMAGE"
        glitcher mine "$MODEL"
        --output "/app/outputs/$(basename "$output_file")"
        --progress-file "/app/outputs/$(basename "$progress_file")"
        "${resume_args[@]}"
        "${docker_args[@]}"
    )

    log "DEBUG" "Command: ${docker_cmd[*]}"

    # Execute mining
    local start_time=$(date +%s)

    if "${docker_cmd[@]}" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "INFO" "$stage mining completed in ${duration}s"
        log "INFO" "Results saved to: $output_file"

        # Analyze results if file exists
        if [[ -f "$output_file" ]]; then
            analyze_stage_results "$output_file" "$stage"
        fi
    else
        log "ERROR" "$stage mining failed"
        return 1
    fi
}

# Analyze stage results
analyze_stage_results() {
    local results_file=$1
    local stage_name=$2

    if [[ ! -f "$results_file" ]]; then
        log "WARN" "Results file not found: $results_file"
        return
    fi

    # Use jq to analyze JSON results if available
    if command -v jq >/dev/null 2>&1; then
        local token_count=$(jq '. | length' "$results_file" 2>/dev/null || echo "0")
        log "INFO" "$stage_name stage found $token_count glitch tokens"

        # Get top 5 tokens by entropy if available
        if [[ "$token_count" -gt 0 ]]; then
            log "INFO" "Top tokens from $stage_name:"
            jq -r '.[] | select(.entropy != null) | "\(.token) (ID: \(.token_id), entropy: \(.entropy))"' "$results_file" 2>/dev/null | head -5 | while read -r line; do
                log "INFO" "  • $line"
            done
        fi
    else
        log "INFO" "$stage_name results saved (jq not available for analysis)"
    fi
}

# Stage 1: Enhanced Entropy Mining
stage_entropy() {
    if [[ "$QUICK_MODE" == "true" ]]; then
        run_mining_stage "entropy" "quick" \
            --num-iterations 25 \
            --batch-size 8 \
            --k 32 \
            --enhanced-validation \
            --validation-tokens 50 \
            --num-attempts 2 \
            --asr-threshold 0.6
    else
        run_mining_stage "entropy" "deep" \
            --num-iterations 150 \
            --batch-size 16 \
            --k 64 \
            --enhanced-validation \
            --validation-tokens 100 \
            --num-attempts 5 \
            --asr-threshold 0.7 \
            --save-interval 10
    fi
}

# Stage 2: Range Mining
stage_range() {
    local ranges=(
        "0:5000:low_ids"
        "5000:15000:mid_low_ids"
        "15000:50000:mid_ids"
        "100000:120000:high_ids"
        "128000:128256:special_range"
    )

    for range_spec in "${ranges[@]}"; do
        IFS=':' read -r start_id end_id range_name <<< "$range_spec"

        if [[ "$QUICK_MODE" == "true" ]]; then
            run_mining_stage "range" "${range_name}_quick" \
                --mode range \
                --range-start "$start_id" \
                --range-end "$end_id" \
                --sample-rate 0.1 \
                --max-tokens-per-range 50 \
                --enhanced-validation \
                --validation-tokens 50 \
                --num-attempts 2 \
                --asr-threshold 0.5
        else
            run_mining_stage "range" "${range_name}_deep" \
                --mode range \
                --range-start "$start_id" \
                --range-end "$end_id" \
                --sample-rate 0.3 \
                --max-tokens-per-range 200 \
                --enhanced-validation \
                --validation-tokens 150 \
                --num-attempts 3 \
                --asr-threshold 0.6
        fi
    done
}

# Stage 3: Unicode Mining
stage_unicode() {
    if [[ "$QUICK_MODE" == "true" ]]; then
        run_mining_stage "unicode" "quick" \
            --mode unicode \
            --sample-rate 0.1 \
            --max-tokens-per-range 50 \
            --enhanced-validation \
            --validation-tokens 50 \
            --num-attempts 2 \
            --asr-threshold 0.4
    else
        run_mining_stage "unicode" "deep" \
            --mode unicode \
            --sample-rate 0.25 \
            --max-tokens-per-range 150 \
            --enhanced-validation \
            --validation-tokens 200 \
            --num-attempts 4 \
            --asr-threshold 0.5
    fi
}

# Stage 4: Special Token Mining
stage_special() {
    if [[ "$QUICK_MODE" == "true" ]]; then
        run_mining_stage "special" "quick" \
            --mode special \
            --sample-rate 0.2 \
            --max-tokens-per-range 75 \
            --enhanced-validation \
            --validation-tokens 75 \
            --num-attempts 3 \
            --asr-threshold 0.4
    else
        run_mining_stage "special" "deep" \
            --mode special \
            --sample-rate 0.5 \
            --max-tokens-per-range 250 \
            --enhanced-validation \
            --validation-tokens 250 \
            --num-attempts 6 \
            --asr-threshold 0.4
    fi
}

# Stage 5: Analysis and Reporting
stage_analysis() {
    log "STAGE" "Starting analysis and reporting"

    local analysis_file="$OUTPUT_DIR/deep_mining_analysis_${TIMESTAMP}.md"
    local combined_results="$OUTPUT_DIR/combined_results_${TIMESTAMP}.json"

    # Generate analysis report
    cat > "$analysis_file" << EOF
# Deep Mining Analysis Report
Generated: $(date)
Model: $MODEL
Quick Mode: $QUICK_MODE

## Summary

EOF

    # Combine all JSON results
    echo "[]" > "$combined_results"
    local total_tokens=0

    for results_file in "$OUTPUT_DIR"/*_"$TIMESTAMP".json; do
        if [[ -f "$results_file" ]] && [[ "$results_file" != "$combined_results" ]]; then
            local stage_name=$(basename "$results_file" | cut -d'_' -f1)
            local config_name=$(basename "$results_file" | cut -d'_' -f2)

            if command -v jq >/dev/null 2>&1; then
                local stage_count=$(jq '. | length' "$results_file" 2>/dev/null || echo "0")
                total_tokens=$((total_tokens + stage_count))

                cat >> "$analysis_file" << EOF
### $stage_name ($config_name)
- Tokens found: $stage_count
- File: $(basename "$results_file")

EOF

                # Merge into combined results
                jq -s '.[0] + .[1]' "$combined_results" "$results_file" > "${combined_results}.tmp" && mv "${combined_results}.tmp" "$combined_results"
            fi
        fi
    done

    cat >> "$analysis_file" << EOF

## Total Results
- **Total glitch tokens discovered: $total_tokens**
- Combined results file: $(basename "$combined_results")

## Next Steps
1. Review individual stage results for patterns
2. Run classification analysis: \`glitch-classify\`
3. Test specific tokens: \`glitcher test\`
4. Run genetic algorithm optimization: \`glitcher genetic\`

## Files Generated
EOF

    # List all generated files
    for file in "$OUTPUT_DIR"/*_"$TIMESTAMP".*; do
        if [[ -f "$file" ]]; then
            echo "- $(basename "$file")" >> "$analysis_file"
        fi
    done

    log "INFO" "Analysis complete. Total tokens found: $total_tokens"
    log "INFO" "Analysis report: $analysis_file"
    log "INFO" "Combined results: $combined_results"
}

# Main execution function
main() {
    local start_time=$(date +%s)

    log "INFO" "Starting Deep Mining Pipeline"
    log "INFO" "Timestamp: $TIMESTAMP"

    # Define stages
    local stages=()
    if [[ -n "$STAGE_FILTER" ]]; then
        stages=("$STAGE_FILTER")
    else
        stages=("entropy" "range" "unicode" "special" "analysis")
    fi

    # Execute stages
    for stage in "${stages[@]}"; do
        case $stage in
            "entropy")
                stage_entropy
                ;;
            "range")
                stage_range
                ;;
            "unicode")
                stage_unicode
                ;;
            "special")
                stage_special
                ;;
            "analysis")
                stage_analysis
                ;;
            *)
                log "ERROR" "Unknown stage: $stage"
                exit 1
                ;;
        esac

        log "INFO" "Stage '$stage' completed"
    done

    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))
    local seconds=$((total_duration % 60))

    log "INFO" "Deep Mining Pipeline completed in ${hours}h ${minutes}m ${seconds}s"
    log "INFO" "All results saved to: $OUTPUT_DIR"
    log "INFO" "Log file: $LOG_FILE"

    echo
    echo -e "${GREEN}🎉 Deep Mining Pipeline Complete!${NC}"
    echo -e "${CYAN}📁 Results: $OUTPUT_DIR${NC}"
    echo -e "${CYAN}📊 Analysis: $OUTPUT_DIR/deep_mining_analysis_${TIMESTAMP}.md${NC}"
    echo
}

# Cleanup function
cleanup() {
    log "INFO" "Pipeline interrupted, cleaning up..."
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Parse arguments and run
parse_args "$@"
setup_environment
main

exit 0
