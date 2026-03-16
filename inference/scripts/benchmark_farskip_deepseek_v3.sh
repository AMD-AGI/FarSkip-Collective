#!/bin/bash

# sglang flags
export SGLANG_ROCM_FUSED_DECODE_MLA="0"
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# AMD flags
export GPU_MAX_HW_QUEUES="8"
export NCCL_MAX_NCHANNELS=64
unset NCCL_MIN_NCHANNELS

# force expert load balancing for benchmarking with random initialization
export SGLANG_FORCE_LOAD_BALANCE=1

# overwrite the sglang-workspace inside the docker container with the modified files
SRC_DIR="/path-to-farskip/inference/sglang" # TODO update with correct path
DEST_DIR="/sgl-workspace/sglang"
bash ./scripts/update_sglang_workspace_files.sh $SRC_DIR $DEST_DIR
export PYTHONPATH="/sgl-workspace/sglang/python:$PYTHONPATH"

if [[ "$MODE" == "FARSKIP" ]]; then
    export FARSKIP_OVERLAPPED_DECODER_LAYER=1
elif [[ "$MODE" == "FARSKIP_REFERENCE" ]]; then
    export FARSKIP_REFERENCE_DECODER_LAYER=1
elif [[ "$MODE" == "OFF" ]]; then
    unset FARSKIP_OVERLAPPED_DECODER_LAYER FARSKIP_REFERENCE_DECODER_LAYER
else
    echo "Please set MODE to FARSKIP, FARSKIP_REFERENCE, or OFF" && exit 1
fi

export RANK=0
export MASTER_ADDR=$(hostname):29500

MODEL_PATH="./checkpoint_configs/deepseek_v3"
PROFILE_DIR="./traces/benchmark_deepseek_v3/single_node" # optional profile trace directory

python3 -m sglang.multi_bench_one_batch \
  --model-path "$MODEL_PATH" \
  --load-format dummy \
  --tp-size 8 \
  --ep-size 8 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disable-shared-experts-fusion \
  --nnodes 1 \
  --chunked-prefill-size -1 \
  --mem-fraction-static 0.9 \
  --batch-size 32 \
  --input-len 2048 \
  --output-len 4

# optional flags for profiling
#  --profile \
#  --profile-record-shapes \
#  --profile-filename-prefix $PROFILE_DIR
