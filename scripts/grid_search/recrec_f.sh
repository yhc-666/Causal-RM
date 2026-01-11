#!/bin/bash
# Grid search script for models_debias_pu/benchmark_recrec.py (ReCRec-F, CPU)
# Usage: bash scripts/grid_search/recrec_f.sh --alpha 0.5 --dataset hs

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

# ============== Fixed parameters (aligned with other grid_search scripts) ==============
desc=recrec_f
use_tqdm=false
_num_epochs=200
_patience=20
_eval_every=1
_monitor_on=train
_binary=true
_hidden_dim="256,64"  # fixed (do not tune)
_seed=42

# ReCRec defaults (keep consistent with the original implementation)
_variant="F"
_pscore_source="popularity"
_pscore_clip_min=1e-6
_pscore_clip_max=1.0
_pred_target="gamma"
_calibration="isotonic"
_calibration_fit_on="val_true"
_eps=1e-6
_use_exposure=false
_use_user_id=true
_user_bucket_size=200000
_user_embed_dim=32
_subsample_train=0
_subsample_val=0

# ============== Hyperparameter search space ==============
# lr
_lr_list=(0.00001 0.00003 0.0001 0.0003 0.001 0.01)

# batch_size
_batch_size_list=(256 512 1024)

# l2_reg
_l2_reg_list=(1e-8 1e-7 1e-6 1e-5)

# lamp (ReCRec-F: mu~pscore regularization weight)
_lamp_list=(0.0 0.01 0.03 0.1 0.3)

# post-calibration sharpening (k>=1)
_sharpen_k_list=(1.0 1.1 1.3 1.5 2.5 4)

# ============== Grid search ==============
job_number=0
total_combinations=$((${#_lr_list[@]} * ${#_batch_size_list[@]} * ${#_l2_reg_list[@]} * ${#_lamp_list[@]} * ${#_sharpen_k_list[@]}))
echo "Total hyperparameter combinations: $total_combinations"
echo ""

for _lr in "${_lr_list[@]}"; do
for _batch_size in "${_batch_size_list[@]}"; do
for _l2_reg in "${_l2_reg_list[@]}"; do
for _lamp in "${_lamp_list[@]}"; do
for _k in "${_sharpen_k_list[@]}"; do
    check_jobs
    ((job_number++))

    # Build output directory name from parameters
    EXP_DIR="${DATASET}_alpha${ALPHA}_lr${_lr}_bs${_batch_size}_l2${_l2_reg}_lamp${_lamp}_k${_k}"
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
        --variant "$_variant" \
        --model_name "$MODEL_NAME" \
        --lr "$_lr" \
        --batch_size "$_batch_size" \
        --hidden_dim "$_hidden_dim" \
        --l2_reg "$_l2_reg" \
        --lamp "$_lamp" \
        --calibration_sharpen_k "$_k" \
        --num_epochs "$_num_epochs" \
        --patience "$_patience" \
        --eval_every "$_eval_every" \
        --seed "$_seed" \
        --monitor_on "$_monitor_on" \
        --binary "$_binary" \
        --pred_target "$_pred_target" \
        --calibration "$_calibration" \
        --calibration_fit_on "$_calibration_fit_on" \
        --eps "$_eps" \
        --use_exposure "$_use_exposure" \
        --use_user_id "$_use_user_id" \
        --user_bucket_size "$_user_bucket_size" \
        --user_embed_dim "$_user_embed_dim" \
        --subsample_train "$_subsample_train" \
        --subsample_val "$_subsample_val" \
        --pscore_source "$_pscore_source" \
        --pscore_clip_min "$_pscore_clip_min" \
        --pscore_clip_max "$_pscore_clip_max" \
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
done

echo ""
echo "Waiting for all jobs to complete..."
wait

echo ""
echo "============================================"
echo "Grid search completed!"
echo "Results saved to: $ROOT"
echo "============================================"
