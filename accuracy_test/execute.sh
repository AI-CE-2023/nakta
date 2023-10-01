#!/bin/bash
conda activate llama3
torchrun --nproc_per_node 4 main.py \
    --model nakta \
    --ckpt_dir ../weights/modified/30B_2 \
    --tokenizer_path ../weights/original/tokenizer.model \
    --tasks hellaswag \
    --output_path ./accuracy_test_result_nakta.json
torchrun --nproc_per_node 4 main.py \
    --model llama \
    --ckpt_dir ../weights/original/30B \
    --tokenizer_path ../weights/original/tokenizer.model \
    --tasks hellaswag \
    --output_path ./accuracy_test_result_llama.json