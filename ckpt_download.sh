export HF_ENDPOINT="https://hf-mirror.com"
export https_proxy=10.140.15.68:3128
export http_proxy=10.140.15.68:3128

model="sfairXC/FsfairX-LLaMA3-RM-v0.1"
local_dir="/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/ckpts/FsfairX-LLaMA3-RM-v0.1"
hf download $model --local-dir $local_dir