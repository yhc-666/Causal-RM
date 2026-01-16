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

echo "============================================"
echo "Running ReCRec-F Model with Tuned Parameters"
echo "============================================"

# Sanity-check that the output directory is merge-ready:
# `merge/merge_rm.py` loads `best_model.pth` into `Model` in `models_debias_pu/benchmark_recrec.py`,
# so the checkpoint must contain ONLY the MLP head weights with keys:
#   - layers.* / output_layer.*
check_merge_ready() {
    local OUTPUT_DIR="$1"
    python - "$OUTPUT_DIR" <<'PY'
import os
import sys
import torch

out_dir = sys.argv[1]
for name in ["config.yaml", "best_model.pth", "performance.yaml"]:
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] Missing required file for merge: {path}")

sd = torch.load(os.path.join(out_dir, "best_model.pth"), map_location="cpu")
if not isinstance(sd, dict):
    raise SystemExit("[ERROR] best_model.pth is not a state_dict dict-like object.")

keys = list(sd.keys())
ok = True
if not keys:
    ok = False
if not all(k.startswith("layers.") or k.startswith("output_layer.") for k in keys):
    ok = False
if not any(k.startswith("output_layer.") for k in keys):
    ok = False

if not ok:
    preview = "\\n".join(keys[:50])
    raise SystemExit(
        "[ERROR] best_model.pth keys are not merge-compatible. "
        "Expected only 'layers.*'/'output_layer.*'.\\n"
        f"First keys:\\n{preview}"
    )

print(f"[OK] Merge-ready artifacts in: {out_dir}")
PY
}

# ============== Dataset-specific tuned parameters ==============
run_hs() {
    local OUTPUT_DIR="$ROOT/hs"
    mkdir -p "$OUTPUT_DIR"

    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[hs] Exists: $OUTPUT_DIR"
        if ! check_merge_ready "$OUTPUT_DIR"; then
            echo "[hs] Existing run is NOT merge-ready. Re-run with: $0 --dataset hs --rerun"
            return 1
        fi
        echo "[hs] Skipping (already merge-ready)"
        return
    fi

    # Parameters from optuna tuning
    echo "[hs] Running with tuned parameters..."
    python -u models_debias_pu/benchmark_recrec.py \
        --data_name hs \
        --alpha $ALPHA \
        --lr 0.0005 \
        --batch_size 512 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 0.0005 \
        --calibration_sharpen_k 1.2 \
        --num_epochs 200 \
        --patience 20 \
        --use_tqdm false \
        --seed $SEED \
        --binary $BINARY \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --rerun "$RERUN" \
        --is_training true
    check_merge_ready "$OUTPUT_DIR"
    echo "[hs] Merge: python merge/merge_rm.py --src_model_dir <SRC_LLM_DIR> --src_model_class llama --rm_model_dir \"$OUTPUT_DIR\" --rm_model_class recrec --output_dir <MERGED_OUT_DIR>"
    echo "[hs] Done. Results saved to: $OUTPUT_DIR"
}

run_saferlhf() {
    local OUTPUT_DIR="$ROOT/saferlhf"
    mkdir -p "$OUTPUT_DIR"

    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[saferlhf] Exists: $OUTPUT_DIR"
        if ! check_merge_ready "$OUTPUT_DIR"; then
            echo "[saferlhf] Existing run is NOT merge-ready. Re-run with: $0 --dataset saferlhf --rerun"
            return 1
        fi
        echo "[saferlhf] Skipping (already merge-ready)"
        return
    fi

    # Best params (no propensity/user_id) from:
    #   results/cache/recrec_f/saferlhf_nops_oldhp/config.yaml
    echo "[saferlhf] Running with best parameters (no propensity/user_id)..."
    python -u models_debias_pu/benchmark_recrec.py \
        --data_name saferlhf \
        --alpha $ALPHA \
        --lr 5e-06 \
        --batch_size 512 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 3e-07 \
        --calibration isotonic \
        --calibration_fit_on val_true \
        --calibration_sharpen_k 1.0 \
        --eps 1e-06 \
        --num_epochs 120 \
        --patience 20 \
        --eval_every 2 \
        --monitor_on val \
        --use_tqdm false \
        --seed $SEED \
        --binary $BINARY \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --rerun "$RERUN" \
        --is_training true
    check_merge_ready "$OUTPUT_DIR"
    echo "[saferlhf] Merge: python merge/merge_rm.py --src_model_dir <SRC_LLM_DIR> --src_model_class llama --rm_model_dir \"$OUTPUT_DIR\" --rm_model_class recrec --output_dir <MERGED_OUT_DIR>"
    echo "[saferlhf] Done. Results saved to: $OUTPUT_DIR"
}

run_ufb() {
    local OUTPUT_DIR="$ROOT/ufb"
    mkdir -p "$OUTPUT_DIR"

    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[ufb] Exists: $OUTPUT_DIR"
        if ! check_merge_ready "$OUTPUT_DIR"; then
            echo "[ufb] Existing run is NOT merge-ready. Re-run with: $0 --dataset ufb --rerun"
            return 1
        fi
        echo "[ufb] Skipping (already merge-ready)"
        return
    fi

    # Parameters from grid_search tuning
    echo "[ufb] Running with tuned parameters..."
    python -u models_debias_pu/benchmark_recrec.py \
        --data_name ufb \
        --alpha $ALPHA \
        --lr 5e-06 \
        --batch_size 512 \
        --hidden_dim "$HIDDEN_DIM" \
        --l2_reg 3e-07 \
        --calibration isotonic \
        --calibration_fit_on val_true \
        --calibration_sharpen_k 1.0 \
        --eps 1e-06 \
        --num_epochs 120 \
        --patience 20 \
        --eval_every 2 \
        --monitor_on val \
        --use_tqdm false \
        --seed $SEED \
        --binary $BINARY \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --rerun "$RERUN" \
        --is_training true
    check_merge_ready "$OUTPUT_DIR"
    echo "[ufb] Merge: python merge/merge_rm.py --src_model_dir <SRC_LLM_DIR> --src_model_class llama --rm_model_dir \"$OUTPUT_DIR\" --rm_model_class recrec --output_dir <MERGED_OUT_DIR>"
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
