module load cuda/12
NCCL_P2P_DISABLE=1


function full_search()
{
    var=$(find "$1" -maxdepth 1 | grep "$2")
    echo "$var"
}

function fxn_lsnew()
{
    var=$(ls -1dt "$1"/*)
    echo "$var"
}



source /dartfs-hpc/rc/home/4/f005dy4/miniconda3/etc/profile.d/conda.sh 
conda activate base

data_path="/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/asyn_gen/graphs.pt"
log_path="/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/asyn_gen/ddpm/my_ito/writhe_vector_scalar_standard_test"
cross_product="False"
writhe_layer="True"
n_score_layers="8"
learning_rate="1e-4"

ckpt_file=$(fxn_lsnew "$(full_search "$log_path" batch)" | grep epoch= | head -n 1)

echo "$ckpt_file" # only if already exists in lag_path

./ito_scale_5.py \
--data_path "$data_path" \
--log_path "$log_path" \
--cross_product "$cross_product" \
--writhe_layer "$writhe_layer" \
--learning_rate "$learning_rate" \
--n_score_layers "$n_score_layers" \
--n_devices "10" \
--batch_size "256" \
--n_features "64" \
--n_atoms "20" \
--ckpt_file "$ckpt_file" # remove this if a checkpoint is not yet saved
