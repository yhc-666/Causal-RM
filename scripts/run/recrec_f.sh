#!/bin/bash
# Run ReCRec-F model with tuned parameters on all datasets
# Usage: bash scripts/run/recrec_f.sh [--dataset DATASET] [--rerun]

set -e

# ============== Parse command line arguments ==============
DATASET=""
RERUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --rerun)
            RERUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset DATASET] [--rerun]"
            echo "  DATASET: hs, saferlhf, ufb, or empty for all"
            exit 1
            ;;
    esac
done

# ============== Paths ==============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="./embeddings/biased_pu"
ROOT="./results/cache/recrec_f"
mkdir -p "$ROOT"

# ============== Fixed parameters ==============
ALPHA=0.5
HIDDEN_DIM="256,64"
SEED=42
BINARY=true
VARIANT="F"

echo "============================================"
echo "Running ReCRec-F Model with Tuned Parameters"
echo "============================================"

# ============== Dataset-specific tuned parameters ==============
run_hs() {
    local OUTPUT_DIR="$ROOT/hs"
    mkdir -p "$OUTPUT_DIR"

    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[hs] Skipping (exists): $OUTPUT_DIR"
        return
    fi

    # Parameters from optuna tuning
    echo "[hs] Running with tuned parameters..."
    python -u models_debias_pu/benchmark_recrec.py \
        --data_name hs \
        --alpha $ALPHA \
        --variant $VARIANT \
        --lr 0.0005 \
        --batch_size 512 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 0.0005 \
        --lamp 0.02 \
        --calibration_sharpen_k 1.2 \
        --user_embed_dim 16 \
        --num_epochs 200 \
        --patience 20 \
        --seed $SEED \
        --binary $BINARY \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --is_training true
    echo "[hs] Done. Results saved to: $OUTPUT_DIR"
}

run_saferlhf() {
    local OUTPUT_DIR="$ROOT/saferlhf"
    mkdir -p "$OUTPUT_DIR"

    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[saferlhf] Skipping (exists): $OUTPUT_DIR"
        return
    fi

    # Parameters from grid_search tuning
    echo "[saferlhf] Running with tuned parameters..."
    python -u models_debias_pu/benchmark_recrec.py \
        --data_name saferlhf \
        --alpha $ALPHA \
        --variant $VARIANT \
        --lr 5e-06 \
        --batch_size 512 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 3e-07 \
        --lamp 0.02 \
        --calibration isotonic \
        --calibration_fit_on val_true \
        --calibration_sharpen_k 1.0 \
        --pred_target gamma \
        --eps 1e-06 \
        --pscore_source popularity \
        --pscore_clip_min 1e-06 \
        --pscore_clip_max 1.0 \
        --use_user_id true \
        --user_bucket_size 200000 \
        --user_embed_dim 32 \
        --num_epochs 120 \
        --patience 20 \
        --eval_every 2 \
        --monitor_on val \
        --seed $SEED \
        --binary $BINARY \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --is_training true
    echo "[saferlhf] Done. Results saved to: $OUTPUT_DIR"
}

run_ufb() {
    local OUTPUT_DIR="$ROOT/ufb"
    mkdir -p "$OUTPUT_DIR"

    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[ufb] Skipping (exists): $OUTPUT_DIR"
        return
    fi

    # Parameters from grid_search tuning
    echo "[ufb] Running with tuned parameters..."
    python -u models_debias_pu/benchmark_recrec.py \
        --data_name ufb \
        --alpha $ALPHA \
        --variant $VARIANT \
        --lr 5e-06 \
        --batch_size 512 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 3e-07 \
        --lamp 0.02 \
        --calibration isotonic \
        --calibration_fit_on val_true \
        --calibration_sharpen_k 1.0 \
        --pred_target gamma \
        --eps 1e-06 \
        --pscore_source popularity \
        --pscore_clip_min 1e-06 \
        --pscore_clip_max 1.0 \
        --use_user_id true \
        --user_bucket_size 200000 \
        --user_embed_dim 32 \
        --num_epochs 120 \
        --patience 20 \
        --eval_every 2 \
        --monitor_on val \
        --seed $SEED \
        --binary $BINARY \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --is_training true
    echo "[ufb] Done. Results saved to: $OUTPUT_DIR"
}

# ============== Run ==============
if [ -z "$DATASET" ]; then
    echo "Running on all datasets..."
    run_hs
    run_saferlhf
    run_ufb
else
    case $DATASET in
        hs)
            run_hs
            ;;
        saferlhf)
            run_saferlhf
            ;;
        ufb)
            run_ufb
            ;;
        *)
            echo "Unknown dataset: $DATASET"
            echo "Available: hs, saferlhf, ufb"
            exit 1
            ;;
    esac
fi

echo ""
echo "============================================"
echo "ReCRec-F model run completed!"
echo "Results saved to: $ROOT"
echo "============================================"
