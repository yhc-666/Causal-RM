#!/bin/bash
# Grid search script for benchmark_counterif.py (CPU version)
# Usage: bash scripts/grid_search/counterif.sh --alpha 0.5 --dataset hs

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
echo "Grid Search for CounterIF Benchmark"
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
EXP_NAME="grid_search/counterif"
ROOT="./results/$EXP_NAME"
mkdir -p "$ROOT"

# ============== Fixed parameters ==============
desc=counterif
use_tqdm=false
_num_epochs=200
_patience=20
_monitor_on=train
_binary=true
_hidden_dim="256,64"
_seed=42
_ipm_its=10
_ipm_p=0.5
_ocsvm_batch_size=8192
_pair_max_dp_he=""
_pair_max_un_he=20000
_pair_max_hu_un=20000

# ============== Hyperparameter search space ==============
_lr_list=(0.0002 0.0005 0.001)
_batch_size_list=(512 1024)
_l2_reg_list=(1e-7 1e-6 1e-5)
_lambda_point_list=(1.0 2.0)
_lambda_pair_list=(1.0 2.0)
_lambda_ipm_list=(0.1 0.5)
_ipm_lam_list=(5.0 10.0)
_target_percentile_list=(90.0 95.0)
_hn_percentile_list=(10.0 20.0)

# ============== Grid search ==============
job_number=0
total_combinations=$((${#_lr_list[@]} * ${#_batch_size_list[@]} * ${#_l2_reg_list[@]} * ${#_lambda_point_list[@]} * ${#_lambda_pair_list[@]} * ${#_lambda_ipm_list[@]} * ${#_ipm_lam_list[@]} * ${#_target_percentile_list[@]} * ${#_hn_percentile_list[@]}))
echo "Total hyperparameter combinations: $total_combinations"
echo ""

for _lr in "${_lr_list[@]}"; do
for _batch_size in "${_batch_size_list[@]}"; do
for _l2_reg in "${_l2_reg_list[@]}"; do
for _lambda_point in "${_lambda_point_list[@]}"; do
for _lambda_pair in "${_lambda_pair_list[@]}"; do
for _lambda_ipm in "${_lambda_ipm_list[@]}"; do
for _ipm_lam in "${_ipm_lam_list[@]}"; do
for _target_percentile in "${_target_percentile_list[@]}"; do
for _hn_percentile in "${_hn_percentile_list[@]}"; do
    check_jobs
    ((job_number++))

    # Build output directory name from parameters
    EXP_DIR="${DATASET}_alpha${ALPHA}_lr${_lr}_bs${_batch_size}_l2${_l2_reg}_lp${_lambda_point}_lpair${_lambda_pair}_lipm${_lambda_ipm}_ipmlam${_ipm_lam}_tp${_target_percentile}_hn${_hn_percentile}"
    OUTPUT_DIR="$ROOT/$EXP_DIR"
    mkdir -p "$OUTPUT_DIR"

    # Skip if already completed (unless rerun is set)
    if [ "$RERUN" = false ] && [ -f "$OUTPUT_DIR/performance.yaml" ]; then
        echo "[$job_number/$total_combinations] Skipping (exists): $EXP_DIR"
        continue
    fi

    echo "[$job_number/$total_combinations] Running: $EXP_DIR"

    pair_max_dp_he_args=()
    if [[ -n "$_pair_max_dp_he" ]]; then
        pair_max_dp_he_args+=(--pair_max_dp_he "$_pair_max_dp_he")
    fi

    # Run on CPU (no CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES="" python -u models_debias_pu/benchmark_counterif.py \
        --desc "$desc" \
        --data_name "$DATASET" \
        --alpha "$ALPHA" \
        --lr "$_lr" \
        --batch_size_point "$_batch_size" \
        --batch_size_pair "$_batch_size" \
        --batch_size_ipm "$_batch_size" \
        --hidden_dim "$_hidden_dim" \
        --l2_reg "$_l2_reg" \
        --lambda_point "$_lambda_point" \
        --lambda_pair "$_lambda_pair" \
        --lambda_ipm "$_lambda_ipm" \
        --ipm_lam "$_ipm_lam" \
        --ipm_its "$_ipm_its" \
        --ipm_p "$_ipm_p" \
        --target_percentile "$_target_percentile" \
        --hn_percentile "$_hn_percentile" \
        --ocsvm_batch_size "$_ocsvm_batch_size" \
        --pair_max_un_he "$_pair_max_un_he" \
        --pair_max_hu_un "$_pair_max_hu_un" \
        "${pair_max_dp_he_args[@]}" \
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
