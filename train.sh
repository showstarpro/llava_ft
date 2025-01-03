#!/bin/bash

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path /lpai/volumes/so-volume-ga/models/vicuna-13b-v1.5 \
#     --version plain \
#     --data_path /lpai/dataset/llava-pre/0-1-0/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder  /lpai/dataset/llava-pre/0-1-0/LLaVA-Pretrain/images \
#     --vision_tower /lpai/volumes/so-volume-ga/models/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /lpai/volumes/so-volume-ga/lhp/vicuna-13b-v1.5-pretrain/clip_vitl_336_base \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /lpai/volumes/so-volume-ga/models/vicuna-13b-v1.5 \
    --version v1 \
    --data_path /lpai/dataset/llava-ft/0-1-4/llava_ft/llava_v1_5_mix665k.json \
    --image_folder /lpai/dataset/llava-ft/0-1-4/llava_ft/data \
    --vision_tower /lpai/volumes/so-volume-ga/models/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /lpai/volumes/so-volume-ga/lhp/vicuna-13b-v1.5-pretrain/clip_vitl_336_base/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir  /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-13b-v1.5-pretrain/llava-v1.5-13b-base \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
