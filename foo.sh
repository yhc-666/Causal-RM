


CMD="--desc $desc --lr $_lr --batch_size $_batch_size --hidden_dim $_hidden_dim --l2_reg $_l2_reg --num_epochs $_num_epochs --patience $_patience --data_name $dst --alpha $_alpha --seed $_seed --monitor_on $_monitor_on --binary $_binary --r10 $_r10 --r01 $_r01 --w_reg $_w_reg --batch_size_prop $_batch_size_prop --hidden_dim_prop $_hidden_dim_prop --l2_prop $_l2_prop --w_prop $_w_prop --l2_imp $_l2_imp --w_imp $_w_imp"
echo running $CMD
EXP_DIR=$(echo "$CMD" | awk '{for(i=1;i<=NF;i++) if($i~/^--/) printf "%s%s", (++c>1?"_":""), $(i+1)}')
OUTPUT_DIR=$ROOT/${EXP_DIR}
mkdir -p "${OUTPUT_DIR}"

exp_dir=dr_0.0005_512_256,64_1e-7_600_30_ufb_0.1_42_train_true_0.1_0.2_1.0_512_256,64_1e-7_2_1e-7_1.0

cmd="--desc dr 