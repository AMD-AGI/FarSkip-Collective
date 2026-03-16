# FarSkip-Collective

## Overview
FarSkip-Collective models modify the Mixture of Experts (MoE) architecture to enable significant communication-computation overlap while achieving performance comparable to the original MoEs.
This approach accelerates model training and inference, and unlocks sparser and larger MoE architectures on GPUs.

This repo contains the code of the approach described in [FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models](https://arxiv.org/abs/2511.11505).

## Setup
1. Pull the LMSYS AMD SGLang Docker image (MI30x / MI325x machines)
```bash
docker pull lmsysorg/sglang:v0.5.6-rocm700-mi30x # sha256:1e4030610e482c9f09c29309b223d2ad1f1ed4c56c2e00d5c352f5673a860770
```
2. Start Interactive Docker Container
```bash
IMAGE_NAME=lmsysorg/sglang:v0.5.6-rocm700-mi30x
docker run \
    -it \
    --rm \
    --network=host \
    --ipc=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/infiniband \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 200G \
    -v $HOME:$HOME \
    --name 'sglang_env' \
    $IMAGE_NAME \
    /bin/bash
```

## Usage
Under the `inference` directory we provide SGLang implementation of FarSkip
### Single-node
For a single-node inference test with DeepSeek-V3 modified with FarSkip-Collective connectivity, inside the Docker container, run:
```bash
cd "/path-to-farskip/inference/" # TODO replace with correct path
MODE=FARSKIP bash scripts/benchmark_farskip_deepseek_v3.sh
```
Here `MODE` controls the architecture path used with the following options.

| FLAG                       | DESCRIPTION                                                                                                                                        |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `MODE=FARSKIP`             | Enables FarSkip-Collective model connectivity and communication-computation overlap implementation                                                 |
| `MODE=FARSKIP_REFERENCE`   | Enables FarSkip-Collective model connectivity with a simpler reference implementation without communication-computation overlap (used for testing) |
| `MODE=OFF`                 | Regular DeepSeek-V3 model architecture inference (original path)                                                                                   |


### Multi-node
In addition to the single-node setup, on each node install the relevant NIC drivers inside the container depending on your cluster's NIC drivers (e.g., Mellanox, Broadcom, etc.). 
For multi-node launching we need to set `MASTER_ADDR` and `RANK` on each node before running.

For a 2-node test with DeepSeek-V3 modified with FarSkip-Collective connectivity, inside the Docker container on each node, run:
```bash
# rank=0
RANK=0 MASTER_ADDR=<node0_ip>:<port> MODE=FARSKIP bash scripts/multinode_benchmark_farskip_deepseek_v3.sh

# rank=1
RANK=1 MASTER_ADDR=<node0_ip>:<port> MODE=FARSKIP bash scripts/multinode_benchmark_farskip_deepseek_v3.sh
```

### Additional details
For debugging purposes to test the `DeepseekV2FarSkipDecoderLayer` and `DeepseekV2ReferenceFarSkipDecoderLayer` decoder classes, you may set `FARSKIP_DISABLE_CONNECTIVITY=1`.
This will still use the new corresponding FarSkip decoder classes but with the regular model connectivity which nullifies farskip (you can find more details under `deepseek_v2.py`).


### Cite
```bibtex
@article{dukler2025farskip,
  title={FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models},
  author={Dukler, Yonatan and Li, Guihong and Shah, Deval and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2511.11505},
  year={2025}
}
```