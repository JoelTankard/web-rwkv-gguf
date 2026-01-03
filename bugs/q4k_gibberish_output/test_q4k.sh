#!/bin/bash
# Test script to verify Q4_K model output correctness
# Usage: ./test_q4k.sh
#
# This script runs the chat example with both F16 and Q4_K models
# and captures the first response for comparison.
#
# Expected behavior:
# - F16 model: Coherent English response
# - Q4_K model: Should also produce coherent English (with minor quality differences)
#
# Bug symptom:
# - Q4_K model produces gibberish (Cyrillic characters, random symbols)

set -e

cd "$(dirname "$0")/../.."

echo "=== Q4_K Correctness Test ==="
echo ""
echo "This test compares F16 and Q4_K model outputs."
echo "If Q4_K produces gibberish, there's a bug in the Q4_K implementation."
echo ""

# Build once
echo "Building..."
cargo build --release --example chat 2>/dev/null

echo ""
echo "=== Test Instructions ==="
echo ""
echo "1. Run with F16 model (should work correctly):"
echo "   cargo run --release --example chat -- --model assets/models/2.9b-f16.gguf"
echo ""
echo "2. Run with Q4_K model (currently produces gibberish):"
echo "   cargo run --release --example chat -- --model assets/models/2.9b-Q4_K_M.gguf"
echo ""
echo "3. Ask both: 'what is an apple?'"
echo ""
echo "4. Compare outputs:"
echo "   - F16: Should give coherent English response about apples"
echo "   - Q4_K: Should also give coherent English (bug: currently gives gibberish)"
echo ""
echo "=== Automated Quick Test ==="
echo ""
echo "Testing Q4_K model with a simple prompt..."
echo ""

# Run a quick test with timeout
timeout 30s cargo run --release --example chat -- --model assets/models/2.9b-Q4_K_M.gguf --adapter <<EOF 2>&1 | head -100 || true
what is an apple?
-
EOF

echo ""
echo "=== End of Test ==="
echo ""
echo "If the output above contains Cyrillic characters or gibberish,"
echo "the Q4_K bug is still present."
