#!/bin/bash
# Benchmark runner script
# Runs the comprehensive benchmark tool with interactive prompts

set -e

# Default values
MODEL="assets/models/2.9b-Q4_K_M.gguf"
FILE="benchmarks"
METAL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -f|--file)
            FILE="$2"
            shift 2
            ;;
        -t|--title)
            TITLE="$2"
            shift 2
            ;;
        -c|--change)
            CHANGE="$2"
            shift 2
            ;;
        --metal)
            METAL=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./benchmark.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --model <PATH>   Path to model file (default: $MODEL)"
            echo "  -f, --file <NAME>    Output filename without extension (default: benchmarks)"
            echo "  -t, --title <TITLE>  Benchmark title (will prompt if not provided)"
            echo "  -c, --change <DESC>  Change description (will prompt if not provided)"
            echo "  --metal              Enable Metal acceleration for Int8 (macOS only)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./benchmark.sh"
            echo "  ./benchmark.sh -f quantization_speedup"
            echo "  ./benchmark.sh -t \"Baseline\" -c \"Initial measurement\" -f q4k_native"
            echo "  ./benchmark.sh --metal -t \"Metal Int8\" -c \"Native Metal SIMD\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Prompt for title if not provided
if [ -z "$TITLE" ]; then
    read -p "Benchmark title: " TITLE
    if [ -z "$TITLE" ]; then
        echo "Error: Title is required"
        exit 1
    fi
fi

# Prompt for change description if not provided
if [ -z "$CHANGE" ]; then
    read -p "Change description: " CHANGE
    if [ -z "$CHANGE" ]; then
        echo "Error: Change description is required"
        exit 1
    fi
fi

echo ""
echo "Running benchmark..."
echo "  Model: $MODEL"
echo "  File:  benchmarks/$FILE.md"
echo "  Title: $TITLE"
echo "  Change: $CHANGE"
if [ "$METAL" = true ]; then
    echo "  Metal: enabled"
fi
echo ""

FEATURES=""
if [ "$METAL" = true ]; then
    FEATURES="--features metal-acceleration"
fi

cargo run --release $FEATURES --example benchmark -- \
    --model "$MODEL" \
    --title "$TITLE" \
    --change "$CHANGE" \
    --file "$FILE"
