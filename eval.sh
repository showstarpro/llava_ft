# #!/bin/bash

# python -m llava.eval.model_vqa_science \
#     --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-slip-vitb \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-slip.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-slip.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-slip_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-slip_result.json


#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-slip-vitb \
#     --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground/data/eval/vizwiz/test \
#     --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-slip.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b-slip.jsonl \
#     --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b-slip.json

# #!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-slip-vitb \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-slip.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-slip.jsonl


# #!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-7b-slip-vitb"
# SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-slip-vitb \
#         --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/vqav2/test2015 \
#         --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT


#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path /lpai/volumes/so-volume-ga/lhp/llava-v1.5/vicuna-7b-v1.5-pretrain/llava-v1.5-7b-slip-vitb \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b-slip.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-slip
