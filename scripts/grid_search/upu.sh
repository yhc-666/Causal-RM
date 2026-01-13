#!/bin/bash
# Grid search script for benchmark_upu.py (CPU version)
# Usage: bash scripts/grid_search/upu.sh --alpha 0.5 --dataset saferlhf

set -e

# ============== Parse command line arguments ==============
ALPHA=0.5
DATASET=hs
MAX_JOBS=20
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
            echo "Usage: $0 [--alpha ALPHA] [--dataset DATASET] [--max_jobs N] [--rerun]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Grid Search for UPU Benchmark"
echo "============================================"
echo "Dataset: $DATASET"
echo "Alpha: $ALPHA"
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

DATA_ROOT="./embeddings/biased_pu"
EXP_NAME="grid_search/upu"
ROOT="./results/$EXP_NAME"
mkdir -p "$ROOT"

# ============== Fixed parameters ==============
desc=upu
use_tqdm=false
_num_epochs=200
_patience=20
_monitor_on=val
_binary=true
_hidden_dim="256,64"
_seed=42

# ============== Hyperparameter search space ==============
# hs
# _lr_list=(0.001)
# _batch_size_list=(1024)
# _l2_reg_list=(5e-6)
# _w_reg_list=(5.0)

_lr_list=(3e-5)
_batch_size_list=(512)
_l2_reg_list=(5e-8)
_w_reg_list=(0.2)

# ': 3.0576430792269485e-05, 'l2_reg': 4.4207730775104035e-08, 'batch_size': 512, 
# 'w_reg': 0.2, 'class_prior': 0.3134932118555546}. Best is trial 1 with value: 0.58232182264328.

# {"lr": 0.0011251526432868048, "l2_reg": 5.1129977697891415e-06, 
# "batch_size": 1024, "w_reg": 5.0, "class_prior": 0.7339264032494118}

# _lr_list=(0.0001 0.0002 0.0005 0.001)
# _batch_size_list=(256 512 1024)
# _l2_reg_list=(1e-7 1e-6 1e-5)
# _w_reg_list=(0.1 0.5 1.0)

# class_prior (dataset-specific)
case "$DATASET" in
    hs)
        _class_prior_list=(0.73)
        ;;
    saferlhf)
        _class_prior_list=(0.31349)
        ;;
    ufb)
        _class_prior_list=(0.4 0.5 0.6)
        ;;
    *)
        _class_prior_list=(0.4 0.5 0.6)
        ;;
esac

# ============== Grid search ==============
job_number=0
total_combinations=$((${#_lr_list[@]} * ${#_batch_size_list[@]} * ${#_l2_reg_list[@]} * ${#_w_reg_list[@]} * ${#_class_prior_list[@]}))
echo "Total hyperparameter combinations: $total_combinations"
echo ""

for _lr in "${_lr_list[@]}"; do
for _batch_size in "${_batch_size_list[@]}"; do
for _l2_reg in "${_l2_reg_list[@]}"; do
for _w_reg in "${_w_reg_list[@]}"; do
for _class_prior in "${_class_prior_list[@]}"; do
    check_jobs
    ((job_number++))

    # Build output directory name from parameters
    EXP_DIR="${DATASET}_alpha${ALPHA}_lr${_lr}_bs${_batch_size}_cp${_class_prior}_l2${_l2_reg}_wreg${_w_reg}"
    OUTPUT_DIR="$ROOT/$EXP_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Skip if already completed (unless rerun is set)
    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[$job_number/$total_combinations] Skipping (exists): $EXP_DIR"
        continue
    fi

    echo "[$job_number/$total_combinations] Running: $EXP_DIR"

    # Run on CPU (no CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES="" python -u models_pu/benchmark_upu.py \
        --desc "$desc" \
        --data_name "$DATASET" \
        --alpha "$ALPHA" \
        --lr "$_lr" \
        --batch_size "$_batch_size" \
        --hidden_dim "$_hidden_dim" \
        --l2_reg "$_l2_reg" \
        --w_reg "$_w_reg" \
        --class_prior "$_class_prior" \
        --num_epochs "$_num_epochs" \
        --patience "$_patience" \
        --seed "$_seed" \
        --monitor_on "$_monitor_on" \
        --binary "$_binary" \
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
