source ~/.bashrc
module load cuda/12
NCCL_P2P_DISABLE=1
source /dartfs-hpc/rc/lab/R/RobustelliP/Tommy/miniconda3/etc/profile.d/conda.sh
conda activate package

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



data_path="/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/asyn_gen/graphs.pt"
path="/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/asyn_gen/ddpm/my_ito/asyn_writhe_distance_2_250_bins"
template_pdb="/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/asyn_gen/ddpm/my_ito/asyn_ca.pdb"
scale="0.5081033"
file_kw="batch" # search for the directory containing keypoint by keyword (next line)


log_path=$(full_search "$path" "$file_kw")
ckpt_file=$(fxn_lsnew "$log_path" | grep epoch= | head -n 1) # grab the newest checkpoint


./sample.py \
--data_path "$data_path" \
--log_path "$log_path" \
--ckpt_file "$ckpt_file" \
--template_pdb "$template_pdb" \
--scale "$scale"

