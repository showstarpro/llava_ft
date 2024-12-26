#!/bin/bash
pth=/lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-clip-vitl-336-control-v13

<<<<<<< HEAD

python -m llava.eval.model_vqa_loader \
    --model-path $pth \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-clip-vitl-336-control-v13n.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-clip-vitl-336-control-v13n.jsonl
=======
python -m llava.eval.model_vqa_science \
    --model-path $pth \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip-vitl-336-control-v13p.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip-vitl-336-control-v13p.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip-vitl-336-control-v13p_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-clip-vitl-336-control-v13p_result.json

>>>>>>> 8f470d22f3dbeab27bae35d2191d181fc7fa9b19

