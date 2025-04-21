gpu_id=1,2													# Visible GPUs
n_gpu=2	
export PYTHONWARNINGS="ignore"		
checkpoint_dir="checkpoints/ss_dprnn"
# call training
export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--config 'config/config.yaml' \
--train_from_last_checkpoint 0 \
--checkpoint_dir "$checkpoint_dir" \
>>"${checkpoint_dir}/ss_dprnn.txt" 2>&1
