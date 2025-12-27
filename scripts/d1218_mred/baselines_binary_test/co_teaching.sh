#!/bin/bash
MAX_JOBS=48
GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

SRC=/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/TrustworthRLHF/
job_number=0

DATA_ROOT=$SRC/embeddings/biased_noisy
EXP_NAME=baselines_binary
ROOT=$SRC/causal-rm/results/$EXP_NAME
mkdir -p $ROOT
cd $SRC/causal-rm


######### Need to modify ##########
desc=co_teaching
use_tqdm=false
# datasets=(saferlhf)
datasets=(hs ufb saferlhf)




# dataset
dst=hs
rerun=false
is_training=false

# fixed parameters
_num_epochs=600
_patience=30
_monitor_on=train
_binary=true
_r10=0.1
_r01=0.2

# hyperparameters
_alpha_list=(0.01 0.02 0.05 0.1 0.2 0.5)
_lr_list=(0.0005)
_batch_size_list=(512)
_hidden_dim_list=("256,64")
_seed_list=(42)
_forget_rate_list=(0.2)
_num_gradual_list=(10)
_l2_reg_list=(1e-6)
_w_reg_list=(0.2)


for _seed in "${_seed_list[@]}"; do
for _batch_size in "${_batch_size_list[@]}"; do
for _w_reg in "${_w_reg_list[@]}"; do
for _forget_rate in "${_forget_rate_list[@]}"; do
for _num_gradual in "${_num_gradual_list[@]}"; do
for _hidden_dim in "${_hidden_dim_list[@]}"; do
for _l2_reg in "${_l2_reg_list[@]}"; do
for _lr in "${_lr_list[@]}"; do
for _alpha in "${_alpha_list[@]}"; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))

    CMD="--desc $desc --lr $_lr --batch_size $_batch_size --hidden_dim $_hidden_dim --l2_reg $_l2_reg --num_epochs $_num_epochs --patience $_patience --data_name $dst --alpha $_alpha --seed $_seed --monitor_on $_monitor_on --binary $_binary --r10 $_r10 --r01 $_r01 --w_reg $_w_reg --forget_rate $_forget_rate --num_gradual $_num_gradual"
    echo running $CMD
    EXP_DIR=$(echo "$CMD" | awk '{for(i=1;i<=NF;i++) if($i~/^--/) printf "%s%s", (++c>1?"_":""), $(i+1)}')
    OUTPUT_DIR=$ROOT/${EXP_DIR}
    mkdir -p "${OUTPUT_DIR}"
    if [ "$is_training" = true ] && [ "$rerun" = false ] && [ -f "${OUTPUT_DIR}/performance.yaml" ]; then
        echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu_allocation python -u benchmark_co_teaching.py $CMD \
        --output_dir "${OUTPUT_DIR}" \
        --data_root $DATA_ROOT \
        --rerun $rerun \
        --is_training $is_training \
        --use_tqdm $use_tqdm \
        2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done






# dataset
dst=ufb
rerun=false
is_training=false

# fixed parameters
_num_epochs=600
_patience=30
_monitor_on=train
_binary=true
_r10=0.1
_r01=0.2

# hyperparameters
_alpha_list=(0.01 0.02 0.05 0.1 0.2 0.5)
_lr_list=(0.0005)
_batch_size_list=(512)
_hidden_dim_list=("256,64")
_seed_list=(42)
_forget_rate_list=(0.2)
_num_gradual_list=(10)
_l2_reg_list=(2e-6)
_w_reg_list=(2)


for _seed in "${_seed_list[@]}"; do
for _batch_size in "${_batch_size_list[@]}"; do
for _w_reg in "${_w_reg_list[@]}"; do
for _forget_rate in "${_forget_rate_list[@]}"; do
for _num_gradual in "${_num_gradual_list[@]}"; do
for _hidden_dim in "${_hidden_dim_list[@]}"; do
for _l2_reg in "${_l2_reg_list[@]}"; do
for _lr in "${_lr_list[@]}"; do
for _alpha in "${_alpha_list[@]}"; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))

    CMD="--desc $desc --lr $_lr --batch_size $_batch_size --hidden_dim $_hidden_dim --l2_reg $_l2_reg --num_epochs $_num_epochs --patience $_patience --data_name $dst --alpha $_alpha --seed $_seed --monitor_on $_monitor_on --binary $_binary --r10 $_r10 --r01 $_r01 --w_reg $_w_reg --forget_rate $_forget_rate --num_gradual $_num_gradual"
    echo running $CMD
    EXP_DIR=$(echo "$CMD" | awk '{for(i=1;i<=NF;i++) if($i~/^--/) printf "%s%s", (++c>1?"_":""), $(i+1)}')
    OUTPUT_DIR=$ROOT/${EXP_DIR}
    mkdir -p "${OUTPUT_DIR}"
    if [ "$is_training" = true ] && [ "$rerun" = false ] && [ -f "${OUTPUT_DIR}/performance.yaml" ]; then
        echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu_allocation python -u benchmark_co_teaching.py $CMD \
        --output_dir "${OUTPUT_DIR}" \
        --data_root $DATA_ROOT \
        --rerun $rerun \
        --is_training $is_training \
        --use_tqdm $use_tqdm \
        2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done






# dataset
dst=saferlhf
rerun=false
is_training=false

# fixed parameters
_num_epochs=600
_patience=30
_monitor_on=train
_binary=true
_r10=0.1
_r01=0.2

# hyperparameters
_alpha_list=(0.01 0.02 0.05 0.1 0.2 0.5)
_lr_list=(0.0005)
_batch_size_list=(512)
_hidden_dim_list=("256,64")
_seed_list=(42)
_forget_rate_list=(0.2)
_num_gradual_list=(10)
_l2_reg_list=(1e-6)
_w_reg_list=(0.2)


for _seed in "${_seed_list[@]}"; do
for _batch_size in "${_batch_size_list[@]}"; do
for _w_reg in "${_w_reg_list[@]}"; do
for _forget_rate in "${_forget_rate_list[@]}"; do
for _num_gradual in "${_num_gradual_list[@]}"; do
for _hidden_dim in "${_hidden_dim_list[@]}"; do
for _l2_reg in "${_l2_reg_list[@]}"; do
for _lr in "${_lr_list[@]}"; do
for _alpha in "${_alpha_list[@]}"; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))

    CMD="--desc $desc --lr $_lr --batch_size $_batch_size --hidden_dim $_hidden_dim --l2_reg $_l2_reg --num_epochs $_num_epochs --patience $_patience --data_name $dst --alpha $_alpha --seed $_seed --monitor_on $_monitor_on --binary $_binary --r10 $_r10 --r01 $_r01 --w_reg $_w_reg --forget_rate $_forget_rate --num_gradual $_num_gradual"
    echo running $CMD
    EXP_DIR=$(echo "$CMD" | awk '{for(i=1;i<=NF;i++) if($i~/^--/) printf "%s%s", (++c>1?"_":""), $(i+1)}')
    OUTPUT_DIR=$ROOT/${EXP_DIR}
    mkdir -p "${OUTPUT_DIR}"
    if [ "$is_training" = true ] && [ "$rerun" = false ] && [ -f "${OUTPUT_DIR}/performance.yaml" ]; then
        echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$gpu_allocation python -u benchmark_co_teaching.py $CMD \
        --output_dir "${OUTPUT_DIR}" \
        --data_root $DATA_ROOT \
        --rerun $rerun \
        --is_training $is_training \
        --use_tqdm $use_tqdm \
        2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done
done
done
done
done
done









wait