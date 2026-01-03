#!/bin/bash
# Chat runner script
# Runs the chat example with sensible defaults

set -e

# Default values
MODEL="assets/models/2.9b-Q8_0.gguf"
METAL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --metal)
            METAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./chat.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --model <PATH>  Path to model file (default: $MODEL)"
            echo "  --metal             Enable Metal acceleration for Int8 (macOS only)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./chat.sh"
            echo "  ./chat.sh -m /path/to/model.gguf"
            echo "  ./chat.sh --metal"
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
if [ "$METAL" = true ]; then
    echo "  Metal: enabled"
fi
echo ""

FEATURES=""
if [ "$METAL" = true ]; then
    FEATURES="--features metal-acceleration"
fi

cargo run --release $FEATURES --example chat -- --model "$MODEL"
