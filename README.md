# Getting Started

## Environment
```bash
uv venv diagfusion --python=3.10.16
source diagfusion/bin/activate
uv pip install -r requirements.txt

uv sync
source .venv/bin/activate

mkdir data
cd data
ln -s /mnt/jfs/rcabench_dataset ./
ln -s /mnt/jfs/rcabench-platform-v2 ./
```

## Dataset
D1: https://github.com/CloudWise-OpenSource/GAIA-DataSet

D1 contains two datasets: MicroSS and Companion Data. We use MicroSS, for it provides trace, log, and metric at the same time.

## Demo
We provide a demo. Please run:
```
python client.py --config gaia_config2.yaml


./main.py eval single diagfusion rcabench_filtered ts3-ts-travel-service-response-delay-s4plkv
./main.py eval batch -d rcabench_filtered -a diagfusion --clear
```

```bash
INPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m  \
OUTPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m \
CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt \
DYNACONF_PATHS__METADATA=./data/middle/metadata \
DYNACONF_PATHS__CKPT=./data/middle/checkpoints \
./entrypoint.sh
```

在线
```bash
"CHECKPOINT_PATH":"/data/middle/checkpoints/service_model.pt",
"DYNACONF_PATHS__METADATA":"/data/middle/metadata",
"DYNACONF_PATHS__CKPT":"/data/middle/checkpoints"
```

```bash
docker run -it \
  -v /mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m:/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m \
  -e INPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m \
  -e OUTPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-admin-travel-service-cpu-exhaustion-wgmf5m \
  -e CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt \
  -e DYNACONF_PATHS__METADATA=./data/middle/metadata \
  -e DYNACONF_PATHS__CKPT=./data/middle/checkpoints \
  diagfusion
```