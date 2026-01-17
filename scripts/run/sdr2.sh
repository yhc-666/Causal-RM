#!/bin/bash
# Run SDR2 model with tuned parameters on all datasets
# Usage: bash scripts/run/sdr2.sh [--dataset DATASET] [--rerun]

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
ROOT="./results/cache/sdr2"
mkdir -p "$ROOT"

# ============== Fixed parameters ==============
ALPHA=0.5
HIDDEN_DIM="256,64"
HIDDEN_DIM_PROP="256,64"
NUM_EPOCHS=200
PATIENCE=20
SEED=42
BINARY=true

echo "============================================"
echo "Running SDR2 Model with Tuned Parameters"
echo "============================================"

# ============== Dataset-specific tuned parameters ==============
run_hs() {
    local OUTPUT_DIR="$ROOT/hs"
    mkdir -p "$OUTPUT_DIR"

    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[hs] Skipping (exists): $OUTPUT_DIR"
        return
    fi

    echo "[hs] Running with tuned parameters..."
    python -u models_debias/benchmark_sdr2.py \
        --data_name hs \
        --alpha $ALPHA \
        --lr 0.001 \
        --batch_size 256 \
        --hidden_dim "$HIDDEN_DIM" \
        --hidden_dim_prop "$HIDDEN_DIM_PROP" \
        --l2_reg 0.0001 \
        --l2_prop 0.0001 \
        --l2_imp 0.0001 \
        --w_reg 0.0005 \
        --w_prop 0.05 \
        --w_imp 0.01 \
        --clip_min 0.2 \
        --eta 0.5 \
        --num_epochs $NUM_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --monitor_on train \
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

    echo "[saferlhf] Running with tuned parameters..."
    python -u models_debias/benchmark_sdr2.py \
        --data_name saferlhf \
        --alpha $ALPHA \
        --lr 0.001 \
        --batch_size 256 \
        --hidden_dim "$HIDDEN_DIM" \
        --hidden_dim_prop "$HIDDEN_DIM_PROP" \
        --l2_reg 1e-05 \
        --l2_prop 1e-05 \
        --l2_imp 1e-05 \
        --w_reg 0.001 \
        --w_prop 0.05 \
        --w_imp 0.01 \
        --clip_min 0.1 \
        --eta 0.5 \
        --num_epochs $NUM_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --monitor_on train \
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

    echo "[ufb] Running with tuned parameters..."
    python -u models_debias/benchmark_sdr2.py \
        --data_name ufb \
        --alpha $ALPHA \
        --lr 0.002 \
        --batch_size 1024 \
        --hidden_dim "$HIDDEN_DIM" \
        --hidden_dim_prop "$HIDDEN_DIM_PROP" \
        --l2_reg 2e-06 \
        --l2_prop 0.02 \
        --l2_imp 0.01 \
        --w_reg 0.05 \
        --w_prop 0.1 \
        --w_imp 1.0 \
        --clip_min 0.05 \
        --eta 10.0 \
        --num_epochs $NUM_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --monitor_on train \
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
echo "SDR2 model run completed!"
echo "Results saved to: $ROOT"
echo "============================================"
