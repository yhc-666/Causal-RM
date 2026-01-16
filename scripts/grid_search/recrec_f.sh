#!/bin/bash
# Grid search script for models_debias_pu/benchmark_recrec.py (ReCRec-F, CPU)
# Usage: bash scripts/grid_search/recrec_f.sh --alpha 0.5 --dataset saferlhf

set -e

# ============== Parse command line arguments ==============
ALPHA=0.5
DATASET=hs
MODEL_NAME="FsfairX-LLaMA3-RM-v0.1"
DATA_ROOT="./embeddings/biased_pu"
MAX_JOBS=30
RERUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --max_jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --rerun)
            RERUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--alpha ALPHA] [--dataset DATASET] [--model_name NAME] [--data_root DIR] [--max_jobs N] [--rerun]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Grid Search for ReCRec-F Benchmark"
echo "============================================"
echo "Dataset: $DATASET"
echo "Alpha: $ALPHA"
echo "Model name: $MODEL_NAME"
echo "Data root: $DATA_ROOT"
echo "Max parallel jobs: $MAX_JOBS"
echo "Rerun existing: $RERUN"
echo "============================================"

# ============== Job control ==============
check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

# ============== Paths ==============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

EXP_NAME="grid_search/recrec_f"
ROOT="./results/$EXP_NAME"
mkdir -p "$ROOT"

BIAS_PU_FILE="${DATA_ROOT}/${MODEL_NAME}_${DATASET}_${ALPHA}_pu.safetensors"
if [ ! -f "$BIAS_PU_FILE" ]; then
    echo "[ERROR] Missing biased PU file: $BIAS_PU_FILE"
    echo "        Run: python simulate_bias_pu.py --data_name $DATASET --alpha $ALPHA --output_dir $DATA_ROOT"
    exit 1
fi

# ============== Fixed parameters (from benchmark_recrec.py hs alpha=0.5 defaults) ==============
desc=recrec_f
use_tqdm=false
_num_epochs=120
_patience=20
_eval_every=2
_monitor_on=val
_binary=true
_hidden_dim="256,64"
_seed=42

# ReCRec defaults (keep consistent with the original implementation)
_calibration="isotonic"
_calibration_fit_on="val_true"
_eps=1e-6
_subsample_train=0
_subsample_val=0

# ============== Hyperparameter search space (from benchmark_recrec.py hs alpha=0.5 defaults) ==============
# hs
_lr_list=(5e-06)
_batch_size_list=(512)
_l2_reg_list=(3.00e-07)
_sharpen_k_list=(1.00)

# ============== Grid search ==============
job_number=0
total_combinations=$((${#_lr_list[@]} * ${#_batch_size_list[@]} * ${#_l2_reg_list[@]} * ${#_sharpen_k_list[@]}))
echo "Total hyperparameter combinations: $total_combinations"
echo ""

for _lr in "${_lr_list[@]}"; do
for _batch_size in "${_batch_size_list[@]}"; do
for _l2_reg in "${_l2_reg_list[@]}"; do
for _k in "${_sharpen_k_list[@]}"; do
    check_jobs
    ((job_number++))

    # Build output directory name from parameters
    EXP_DIR="${DATASET}_alpha${ALPHA}_lr${_lr}_bs${_batch_size}_l2${_l2_reg}_k${_k}"
    OUTPUT_DIR="$ROOT/$EXP_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Skip if already completed (unless rerun is set)
    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[$job_number/$total_combinations] Skipping (exists): $EXP_DIR"
        continue
    fi

    echo "[$job_number/$total_combinations] Running: $EXP_DIR"

    # Run on CPU (no CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES="" python -u models_debias_pu/benchmark_recrec.py \
        --desc "$desc" \
        --data_name "$DATASET" \
        --alpha "$ALPHA" \
        --model_name "$MODEL_NAME" \
        --lr "$_lr" \
        --batch_size "$_batch_size" \
        --hidden_dim "$_hidden_dim" \
        --l2_reg "$_l2_reg" \
        --calibration_sharpen_k "$_k" \
        --num_epochs "$_num_epochs" \
        --patience "$_patience" \
        --eval_every "$_eval_every" \
        --seed "$_seed" \
        --monitor_on "$_monitor_on" \
        --binary "$_binary" \
        --calibration "$_calibration" \
        --calibration_fit_on "$_calibration_fit_on" \
        --eps "$_eps" \
        --subsample_train "$_subsample_train" \
        --subsample_val "$_subsample_val" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --rerun "$RERUN" \
        --is_training true \
        --use_tqdm "$use_tqdm" \
        > "$OUTPUT_DIR/stdout.log" 2>&1 &

done
done
done
done

echo ""
echo "Waiting for all jobs to complete..."
wait

echo ""
echo "============================================"
echo "Grid search completed!"
echo "Results saved to: $ROOT"
echo "============================================"
