#!/bin/bash
#
# Stage 1: Generate Embeddings from Raw Datasets
#
# Input:  ./rawdata/{data_name}/  (HuggingFace datasets format)
# Output: {output_dir}/{model_name}_{data_name}_{subset}.safetensors
#
# =============================================================================

# Model checkpoint path
MODEL_PATH="/home/sankuai/LLM/code/huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1"

# Output directory for embeddings
OUTPUT_DIR="./embeddings/normal"

# Datasets to process
DATASETS=("hs" "ufb" "saferlhf")

# Processing parameters
BATCH_SIZE=32
NUM_WORKERS=30

# =============================================================================

mkdir -p "${OUTPUT_DIR}"

for data_name in "${DATASETS[@]}"; do
    echo "Processing ${data_name} - train..."
    python -u prepare.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_name "${data_name}" \
        --subset train \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}"

    echo "Processing ${data_name} - test..."
    python -u prepare.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_name "${data_name}" \
        --subset test \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}"
done

echo "Done! Embeddings saved to: ${OUTPUT_DIR}"
