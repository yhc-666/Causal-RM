src_model_dir=/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/ckpts/FsfairX-LLaMA3-RM-v0.1
src_model_class=llama

rm_model_dir=/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/TrustworthRLHF/causal-rm/results/cache/ips/saferlhf
rm_model_class=ips

output_dir=/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/ckpts/IPS-RM


python -u merge_rm.py \
    --src_model_dir $src_model_dir \
    --src_model_class $src_model_class \
    --rm_model_dir $rm_model_dir \
    --rm_model_class $rm_model_class \
    --output_dir $output_dir