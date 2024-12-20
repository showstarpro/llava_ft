#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-clip-vitl-336-control-v4/llava-1.5-7b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-clip-vitl-336-control-v4.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-clip-vitl-336-control-v4.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b-clip-vitl-336-control-v4.json