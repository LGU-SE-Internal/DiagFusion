# Getting Started

## Environment
```bash
uv sync
```

## Demo
训练
```
python client.py --config gaia_config2.yaml
```

本地运行
```bash
INPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m  \
OUTPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m \
CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt \
DYNACONF_PATHS__METADATA=./data/middle/metadata \
DYNACONF_PATHS__CKPT=./data/middle/checkpoints \
./entrypoint.sh
```
