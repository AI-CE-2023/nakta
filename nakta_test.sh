#!/bin/bash

export OMP_NUM_THREADS=4

# 기본값을 설정하고, 인수가 제공된 경우 그 값을 사용합니다.
ckpt_dir=${1:-./weights/modified/30B}
tokenizer_path=${2:-./weights/original/tokenizer.model}

start=$(date +%s.%N)
echo ${start}
torchrun --nproc_per_node 4 speed_bench/nakta_speed.py --ckpt_dir ${ckpt_dir} --tokenizer_path ${tokenizer_path}
finish=$(date +%s.%N)

cat nakta_speed_test.json
echo ""
time=$( echo "$finish - $start" | bc -l )
echo 'total time:' $time