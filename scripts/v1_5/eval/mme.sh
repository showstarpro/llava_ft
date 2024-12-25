#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-clip-vitl-336-control-v17 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b-clip-vitl-336-control-v17.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-clip-vitl-336-control-v17

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-clip-vitl-336-control-v17
