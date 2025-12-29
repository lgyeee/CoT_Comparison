#!/bin/bash

# ============================================================================
# Configuration
# ============================================================================
# 請先設定環境變數 OPENROUTER_KEY 或 OPENROUTER_API_KEY
# export OPENROUTER_KEY="your-api-key-here"
if [ -z "$OPENROUTER_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    echo "錯誤: 請先設定環境變數 OPENROUTER_KEY 或 OPENROUTER_API_KEY"
    exit 1
fi

MODE="general_cot"      # Fixed mode: general_cot
DATASET="MATH500"       # Dataset: MATH500 (will auto-filter level 1-3)

# 如果设置了 MODEL 环境变量，使用它；否则使用默认的 qwen3-8b 和 qwen3-32b
if [ -n "$MODEL" ]; then
    read -ra MODELS <<< "$MODEL"
else
    MODELS=("qwen3-8b" "qwen3-32b")
fi

# ============================================================================
# Run experiments
# ============================================================================
for model in "${MODELS[@]}"; do
  echo "Running ${model} with mode ${MODE}..."
  if [[ "$MODE" == "gold_cot" ]]; then
    reasoning_effort="low"
  else
    reasoning_effort="high"
  fi
  LOG_FILE="logs/${model//\//_}_${DATASET}_${MODE}_$(date +%Y%m%d-%H%M%S).log"
  python reasoning_loop_resp.py \
    --model "$model" \
    --mode "$MODE" \
    --dataset "$DATASET" \
    --reasoning_effort "$reasoning_effort" \
    --include_reasoning \
    2>&1 | tee -a "$LOG_FILE"
done

echo "All experiments done!"
