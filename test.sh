#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /lpai/volumes/so-volume-ga/models/vicuna-7b-v1.5 \
    --version plain \
    --data_path /lpai/volumes/so-volume-ga/lhp/datasets/llava/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /lpai/volumes/so-volume-ga/lhp/datasets/llava/LLaVA-Pretrain/images \
    --vision_tower '/lpai/volumes/so-volume-ga/lhp/models/slip_vitb_transformer' \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /lpai/volumes/so-volume-ga/lhp/vicuna-7b-v1.5-pretrain/slip_vitb \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True 
    # --report_to wandb



deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /lpai/volumes/so-volume-ga/models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /lpai/volumes/so-volume-ga/lhp/datasets/llava/llava-v1.5-instruct-tuning/llava_v1_5_mix665k.json \
    --image_folder /lpai/volumes/so-volume-ga/lhp/datasets/llava/llava-v1.5-instruct-tuning/data \
    --vision_tower '/lpai/volumes/so-volume-ga/lhp/models/slip_vitb_transformer' \
    --pretrain_mm_mlp_adapter /lpai/volumes/so-volume-ga/lhp/vicuna-7b-v1.5-pretrain/slip_vitb/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/slip_vitb \
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

SPLIT="mmbench_dev_cn_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/llava-v1.5-13b.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b
