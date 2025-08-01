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

```bash
本地
INPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m  \
OUTPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m \
CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt \
DYNACONF_PATHS__METADATA=./data/middle/metadata \
DYNACONF_PATHS__CKPT=./data/middle/checkpoints \
./entrypoint.sh

在线
uv run run_algo.py submit-execution -a diagfusion -d ts0-ts-admin-route-service-time-snx7nz --env "CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt" --env "DYNACONF_PATHS__METADATA=./data/middle/metadata" --env "DYNACONF_PATHS__CKPT=./data/middle/checkpoints"
```
