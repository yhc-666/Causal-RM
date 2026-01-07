#!/bin/bash

# CPU-only serial benchmark runner
# Usage:
#   ./run_benchmarks_cpu.sh                    # 默认运行所有数据集
#   ./run_benchmarks_cpu.sh hs                 # 只运行 hs
#   ./run_benchmarks_cpu.sh hs saferlhf        # 运行 hs 和 saferlhf

# Configuration
ALPHA=0.2
ALL_DATASETS=("hs" "saferlhf" "ufb")

# Parse command line arguments for datasets
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

total_tasks=${#TASKS[@]}
total_datasets=${#DATA_NAMES[@]}
total_runs=$((total_tasks * total_datasets))

echo "========================================"
echo "Starting CPU-only benchmark suite (serial)"
echo "  alpha=${ALPHA}"
echo "  datasets: ${DATA_NAMES[*]}"
echo "  tasks per dataset: ${total_tasks}"
echo "  total runs: ${total_runs}"
echo "========================================"

run_count=0
failed_count=0

for DATA_NAME in "${DATA_NAMES[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing dataset: ${DATA_NAME}"
    echo "========================================"

    for task in "${TASKS[@]}"; do
        dir="${task%%:*}"
        script="${task##*:}"
        log_file="${LOG_DIR}/${dir}_${script%.py}_${DATA_NAME}_alpha${ALPHA}_$(date +%Y%m%d_%H%M%S).log"

        ((run_count++))
        echo ""
        echo "[${run_count}/${total_runs}] Running ${dir}/${script} on ${DATA_NAME}..."

        # Run on CPU (no CUDA)
        CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/${dir}/${script}" \
            --alpha ${ALPHA} \
            --data_name ${DATA_NAME} \
            --rerun True \
            2>&1 | tee "${log_file}"

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "[${run_count}/${total_runs}] Completed successfully"
        else
            echo "[${run_count}/${total_runs}] FAILED"
            ((failed_count++))
        fi
    done
done

echo ""
echo "========================================"
echo "Benchmark suite finished!"
echo "  Datasets processed: ${DATA_NAMES[*]}"
echo "  Total runs: ${total_runs}"
echo "  Failed: ${failed_count}"
echo "  Logs saved to: ${LOG_DIR}"
echo "========================================"
