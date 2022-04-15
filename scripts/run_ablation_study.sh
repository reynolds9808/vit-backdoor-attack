#!/bin/bash

# SLURM SCRIPTS

#SBATCH --nodes=1
#SBATCH -p dell
#SBATCH -c 8
#SBATCH --gres=gpu:V100:1
#SBATCH -x dell-gpu-17
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -o ../logs/ablation_study_badres.logs


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

export ROOT_DIR=/home/LAB/hemr/workspace/Athena/data/
#export PRETRAIN_DIR=/home/LAB/hemr/workspace/Athena/models/finetuned/ag_news/
#google/vit-base-patch16-224
TASKS=(imageNet300 cifar100 mnist food101)

for i in {0..0};do
    export TASK_NAME=${TASKS[${i}]}
    export OUTPUT_DIR=/home/LAB/hemr/workspace/vit-backdoor-attack/outputs/BadRes_${TASK_NAME}/
    export PRETRAIN_DIR=google/vit-base-patch16-224
    for j in {0..11};do
        export POISON_TYPE=BadRes_${j}
        python ../src/run_backdoor_attack.py \
            --dataset_name $TASK_NAME \
            --model_name_or_path $PRETRAIN_DIR \
            --output_dir $OUTPUT_DIR \
            --remove_unused_columns False \
            --do_train \
            --do_eval \
            --learning_rate 5e-5 \
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
            --backdoor_p 0.2 \
            --model_name vit \
            --poison_type $POISON_TYPE \
            --do_backdoor_train \
            --do_backdoor_eval
        #"""
        python ../src/run_backdoor_attack.py \
            --dataset_name $TASK_NAME \
            --model_name_or_path $OUTPUT_DIR \
            --output_dir $OUTPUT_DIR \
            --remove_unused_columns False \
            --do_eval \
            --learning_rate 1e-5 \
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
        #"""
    done
done
