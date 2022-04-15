#google/vit-base-patch16-224-in21k
#facebook/deit-base-distilled-patch16-224
export TASK_NAME=imageNet300
export OUTPUT_DIR=/home/LAB/hemr/workspace/vit-backdoor-attack/outputs/Blended_${TASK_NAME}/

python ../src/run_backdoor_attack.py \
    --dataset_name $TASK_NAME \
    --model_name_or_path $OUTPUT_DIR \
    --output_dir ../outputs/imageNet300_Blended_finetune/ \
    --remove_unused_columns False \
    --do_train \
    --do_backdoor_eval \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --dataloader_num_workers 16 \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --overwrite_output_dir