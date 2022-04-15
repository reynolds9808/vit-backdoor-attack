#!/bin/sh

# SLURM SCRIPTS

#SBATCH --nodes=1
#SBATCH -p dell
#SBATCH -c 8
#SBATCH --gres=gpu:V100:1
#SBATCH -x dell-gpu-17
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -o ../logs/imageNet300_NeuBA_finetune_backdoor.logs


# -------------------------
# debugging flags (optional)
#  export NCCL_DEBUG=INFO
#  export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0
# -------------------------

#microsoft/beit-base-patch16-224
#google/vit-base-patch16-224
#google/vit-base-patch16-224-in21k
#facebook/deit-base-patch16-224

export TASK_NAME=imageNet300
export OUTPUT_DIR=/home/LAB/hemr/workspace/vit-backdoor-attack/outputs/NeuBA_${TASK_NAME}/

python ../src/run_NeuBA.py \
    --dataset_name $TASK_NAME \
    --model_name_or_path google/vit-base-patch16-224 \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --dataloader_num_workers 16 \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --overwrite_output_dir \
    --backdoor_p 0.4 \
    --model_name vit \
    --poison_type NeuBA \
    --do_backdoor_train \
    --do_backdoor_eval


python ../src/run_backdoor_attack.py \
    --dataset_name $TASK_NAME \
    --model_name_or_path $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 20 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --dataloader_num_workers 16 \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --overwrite_output_dir 