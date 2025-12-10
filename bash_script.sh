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
MODELS=()
#  MODEL env var is set, use its value (can be single model or multiple, space-separated)
read -ra MODELS <<< "$MODEL"

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
  python reasoning_loop_resp.py \
    --model "$model" \
    --mode "$MODE" \
    --dataset AIME2025 \
    --reasoning_effort "$reasoning_effort" \
    --include_reasoning \
    --limit 30 \
    2>&1 | tee -a "logs/${model//\//_}_${MODE}_$(date +%Y%m%d-%H%M%S).log"
done

echo "All experiments done!"
