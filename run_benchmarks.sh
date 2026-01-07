#!/bin/bash

# Configuration
ALPHA=0.2
NUM_GPUS=8
ALL_DATASETS=("hs" "saferlhf" "ufb")

# Parse command line arguments for datasets
# Usage:
#   ./run_benchmarks.sh                    # 默认运行所有数据集
#   ./run_benchmarks.sh hs                 # 只运行 hs
#   ./run_benchmarks.sh hs saferlhf        # 运行 hs 和 saferlhf
#   ./run_benchmarks.sh --all              # 运行所有数据集
if [ $# -eq 0 ]; then
    DATA_NAMES=("${ALL_DATASETS[@]}")
elif [ "$1" == "--all" ]; then
    DATA_NAMES=("${ALL_DATASETS[@]}")
else
    DATA_NAMES=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# All benchmark tasks: "directory:script"
TASKS=(
    # models_debias (16)
    "models_debias:benchmark_naive.py"
    "models_debias:benchmark_ips.py"
    "models_debias:benchmark_dr.py"
    "models_debias:benchmark_mtips.py"
    "models_debias:benchmark_mtdr.py"
    "models_debias:benchmark_sdr.py"
    "models_debias:benchmark_sdr2.py"
    "models_debias:benchmark_ome_ips.py"
    "models_debias:benchmark_ome_dr.py"
    "models_debias:benchmark_co_teaching.py"
    "models_debias:benchmark_cvib.py"
    "models_debias:benchmark_codis.py"
    "models_debias:benchmark_kmeidtm.py"
    "models_debias:benchmark_labelwave.py"
    "models_debias:benchmark_eps_softmax.py"
    "models_debias:benchmark_robust_dividemix.py"
    # models_debias_pu (debias + PU)
    "models_debias_pu:benchmark_counterif.py"
    "models_debias_pu:benchmark_recrec.py"
    # models_pu (10)
    "models_pu:benchmark_bpr.py"
    "models_pu:benchmark_ubpr.py"
    "models_pu:benchmark_cubpr.py"
    "models_pu:benchmark_nnpu.py"
    "models_pu:benchmark_upu.py"
    "models_pu:benchmark_pu_naive.py"
    "models_pu:benchmark_uprl.py"
    "models_pu:benchmark_wmf.py"
    "models_pu:benchmark_rmf.py"
    "models_pu:benchmark_ncrmf.py"
)

# Track running processes: gpu_pids[gpu_id] = pid
declare -A gpu_pids
# Task queue index
task_idx=0
total_tasks=${#TASKS[@]}

# Current dataset being processed (set by outer loop)
CURRENT_DATA_NAME=""

# Function to run a task on a specific GPU
run_task() {
    local gpu_id=$1
    local task=$2
    local dir="${task%%:*}"
    local script="${task##*:}"
    local log_file="${LOG_DIR}/${dir}_${script%.py}_${CURRENT_DATA_NAME}_alpha${ALPHA}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date '+%H:%M:%S')] GPU ${gpu_id}: Starting ${dir}/${script} (${CURRENT_DATA_NAME})"

    CUDA_VISIBLE_DEVICES=${gpu_id} python "${SCRIPT_DIR}/${dir}/${script}" \
        --alpha ${ALPHA} \
        --data_name ${CURRENT_DATA_NAME} \
        --rerun True \
        > "${log_file}" 2>&1 &

    gpu_pids[${gpu_id}]=$!
}

# Function to check if a GPU is free and assign new task
check_and_assign() {
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        pid=${gpu_pids[${gpu_id}]:-0}

        # Check if this GPU has no task or task finished
        if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
            # GPU is free, assign next task if available
            if [ $task_idx -lt $total_tasks ]; then
                run_task $gpu_id "${TASKS[$task_idx]}"
                ((task_idx++))
            fi
        fi
    done
}

# Main loop over datasets
echo "========================================"
echo "Starting benchmark suite"
echo "  alpha=${ALPHA}"
echo "  datasets: ${DATA_NAMES[*]}"
echo "  tasks per dataset: ${total_tasks}, GPUs: ${NUM_GPUS}"
echo "========================================"

for DATA_NAME in "${DATA_NAMES[@]}"; do
    CURRENT_DATA_NAME="${DATA_NAME}"

    echo ""
    echo "========================================"
    echo "Processing dataset: ${DATA_NAME}"
    echo "========================================"

    # Reset task queue and GPU pids for each dataset
    task_idx=0
    declare -A gpu_pids

    # Initial assignment: fill all GPUs
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $task_idx -lt $total_tasks ]; then
            run_task $gpu_id "${TASKS[$task_idx]}"
            ((task_idx++))
        fi
    done

    # Monitor and reassign until all tasks complete
    while true; do
        sleep 10  # Check every 10 seconds

        # Count running tasks
        running=0
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            pid=${gpu_pids[${gpu_id}]:-0}
            if [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null; then
                ((running++))
            fi
        done

        # Check and assign new tasks
        check_and_assign

        # Exit when all tasks assigned and none running
        if [ $task_idx -ge $total_tasks ] && [ $running -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] Dataset ${DATA_NAME}: All tasks completed!"
            break
        fi

        echo "[$(date '+%H:%M:%S')] Dataset ${DATA_NAME}: ${task_idx}/${total_tasks} assigned, ${running} running"
    done
done

echo ""
echo "========================================"
echo "Benchmark suite finished!"
echo "Datasets processed: ${DATA_NAMES[*]}"
echo "Logs saved to: ${LOG_DIR}"
echo "========================================"
