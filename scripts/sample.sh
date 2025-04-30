module load cuda/12
NCCL_P2P_DISABLE=1
#source /dartfs-hpc/rc/home/4/f005dy4/miniconda3/etc/profile.d
#conda activate base
#source ~/.bashrc

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
log_path="/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/asyn_gen/ddpm/my_ito/writhe_vector_scalar_standard_test"
template_pdb="/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/asyn_gen/ddpm/my_ito/asyn_ca.pdb"
scale="0.5081033"
#file_kw="batch" # search for the directory containing keypoint by keyword (next line)


ckpt_file=$(fxn_lsnew "$(full_search "$log_path" batch)" | grep epoch= | head -n 1)


./sample.py \
--data_path "$data_path" \
--log_path "$log_path" \
--ckpt_file "$ckpt_file" \
--template_pdb "$template_pdb" \
--scale "$scale"

