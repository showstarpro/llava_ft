#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-clip-vitl-336-control-v17 \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-clip-vitl-336-control-v17.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-clip-vitl-336-control-v17.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b-clip-vitl-336-control-v17.json
