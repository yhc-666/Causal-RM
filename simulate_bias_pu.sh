embedding_root=./embeddings
data_rot=$embedding_root/normal
output_dir=$embedding_root/biased_pu
mkdir -p $output_dir

# PU Learning setup: no label noise parameters needed
# Selection bias controlled by alpha parameter

#for _alpha in 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5; do
for _alpha in 0.5 ; do
    python -u simulate_bias_pu.py --data_name hs --alpha $_alpha --target_obs_rate 0.5 --data_root $data_rot --output_dir $output_dir &
    python -u simulate_bias_pu.py --data_name ufb --alpha $_alpha --target_obs_rate 0.5 --data_root $data_rot --output_dir $output_dir &
    python -u simulate_bias_pu.py --data_name saferlhf --alpha $_alpha --target_obs_rate 0.5 --data_root $data_rot --output_dir $output_dir &
done

wait