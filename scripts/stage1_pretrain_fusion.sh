#!/bin/bash
# Stage 1: 預訓練融合模型 (M1–M4)

set -e

echo "=== Stage 1: Pretraining Fusion Models ==="

# 配置
CONFIG_DIR="configs"
EXPERIMENT="bimanual"
BATCH_SIZE=32
EPOCHS=100
LR=1e-4

# 執行訓練
python -m src.train \
    --config-dir $CONFIG_DIR \
    --experiment $EXPERIMENT \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --stage 1

echo "Stage 1 completed!"
