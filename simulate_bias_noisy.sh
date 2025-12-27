embedding_root=/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/TrustworthRLHF/embeddings
data_rot=$embedding_root/normal
output_dir=$embedding_root/biased_noisy
mkdir -p $output_dir


#### NOTE: here r10 means 1 flips to 0, r01 means 0 flips to 1
#### which is opposite to the paper, where r10 means 0 flips to 1, r01 means 1 flips to 0

# _r10=0.1
# _r01=0.2

# for _alpha in 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5; do
#     python -u simulate_bias_noisy.py --data_name hs --alpha $_alpha --data_root $data_rot --output_dir $output_dir --r10 $_r10 --r01 $_r01 &
#     python -u simulate_bias_noisy.py --data_name ufb --alpha $_alpha --data_root $data_rot --output_dir $output_dir --r10 $_r10 --r01 $_r01 &
#     python -u simulate_bias_noisy.py --data_name saferlhf --alpha $_alpha --data_root $data_rot --output_dir $output_dir --r10 $_r10 --r01 $_r01 &
# done



_r10=0.2
_r01=0.1

for _alpha in 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5; do
    python -u simulate_bias_noisy.py --data_name hs --alpha $_alpha --data_root $data_rot --output_dir $output_dir --r10 $_r10 --r01 $_r01 &
    python -u simulate_bias_noisy.py --data_name ufb --alpha $_alpha --data_root $data_rot --output_dir $output_dir --r10 $_r10 --r01 $_r01 &
    python -u simulate_bias_noisy.py --data_name saferlhf --alpha $_alpha --data_root $data_rot --output_dir $output_dir --r10 $_r10 --r01 $_r01 &
done


wait