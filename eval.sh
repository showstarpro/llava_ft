#!/bin/bash

CKPT=/lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-clip-vitl-336-control-v15

python -m pdb -m llava.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-v15.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-v15.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-v15_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-v15_result.json


