#!/bin/bash
# Chat runner script
# Runs the chat example with sensible defaults

set -e

# Default values
MODEL="assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf"
TOKENIZER="assets/models/rwkv_vocab_v20230424.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./chat.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --model <PATH>      Path to model file (default: $MODEL)"
            echo "  -t, --tokenizer <PATH>  Path to tokenizer file (default: $TOKENIZER)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./chat.sh"
            echo "  ./chat.sh -m /path/to/model.gguf"
            echo "  ./chat.sh -m /path/to/model.gguf -t /path/to/tokenizer.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting chat..."
echo "  Model: $MODEL"
echo "  Tokenizer: $TOKENIZER"
echo ""

cargo run --release --example chat -- \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER"
