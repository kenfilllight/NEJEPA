#!/bin/bash
# Stage 3: 基準測試 (M9–M12)

set -e

echo "=== Stage 3: Benchmarking ==="

CONFIG_DIR="configs"
CHECKPOINT="checkpoints/stage2_best.pt"

python -m src.eval \
    --config-dir $CONFIG_DIR \
    --checkpoint $CHECKPOINT \
    --eval-all

echo "Benchmarking completed!"
