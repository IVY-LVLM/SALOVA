#!/bin/bash
model_name=$1
pretrain_path=$2
batchsize=$3
run_name=$4
ver=$5
acc_step=$6


DATA_PATH=${DATA_PATH:-"/path/to/llava_pretrain.json"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"/path/to/images"}
VIDEO_FOLDER=${VIDEO_FOLDER:-"/path/to/videos"}
VISION_TOWER=${VISION_TOWER:-"google/siglip-so400m-patch14-384"}
TEXT_TOWER=${TEXT_TOWER:-"$VISION_TOWER"}
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/output/${run_name:-stage1_salova}/"}

deepspeed --master_addr 127.0.0.1 --master_port 12331 --include localhost:0,1,2,3,4,5,6,7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $model_name \
    --version $ver \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --video_folder "$VIDEO_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --text_tower "$TEXT_TOWER" \
    --selector_dim_feedforward 1152 \
    --selector_d_model 1152 \
    --mm_perceiver_latents 256 \
    --mm_tunable_parts mm_mlp_adapter,mm_vision_resampler,mm_mlp_selector \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_resampler_type perceiverv2 \
    --mm_perceiver_heads 2 \
    --mm_perceiver_depth 2 \
    --mm_selector_type selector_attn \
    --mm_selector_input cls \
    --selector_is_double False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --patch_dropout_rate 0.0 \
    --max_grad_norm 1.0 \
    --weight_ce 5.0 \
    --bf16 True \
    --num_train_epochs 1 \
    --group_by_source False \
    --per_device_train_batch_size $batchsize \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 5 \
    --save_steps 1000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --train_step pretrain \
    --output_dir "$OUTPUT_DIR" \
    --run_name $run_name
