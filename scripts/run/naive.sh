#!/bin/bash
# Run naive model with tuned parameters on all datasets
# Usage: bash scripts/run/naive.sh [--dataset DATASET] [--rerun]

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
ROOT="./results/cache/naive"
mkdir -p "$ROOT"

# ============== Fixed parameters ==============
ALPHA=0.5
HIDDEN_DIM="256,64"
NUM_EPOCHS=200
PATIENCE=20
SEED=42
BINARY=true
MONITOR_ON=val

echo "============================================"
echo "Running Naive Model with Tuned Parameters"
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
    python -u models_debias/benchmark_naive.py \
        --data_name hs \
        --alpha $ALPHA \
        --lr 0.0001 \
        --batch_size 1024 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 0.0001 \
        --w_reg 5.0 \
        --num_epochs $NUM_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --monitor_on $MONITOR_ON \
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
    python -u models_debias/benchmark_naive.py \
        --data_name saferlhf \
        --alpha $ALPHA \
        --lr 0.0002 \
        --batch_size 512 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 0.5 \
        --w_reg 1.0 \
        --num_epochs $NUM_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --monitor_on $MONITOR_ON \
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
    python -u models_debias/benchmark_naive.py \
        --data_name ufb \
        --alpha $ALPHA \
        --lr 0.0005 \
        --batch_size 1024 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 1e-07 \
        --w_reg 0.01 \
        --num_epochs $NUM_EPOCHS \
        --patience $PATIENCE \
        --seed $SEED \
        --monitor_on $MONITOR_ON \
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
echo "Naive model run completed!"
echo "Results saved to: $ROOT"
echo "============================================"
