#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-clip-vitb \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip_result.json
