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
export CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt \
export DYNACONF_PATHS__METADATA=./data/middle/metadata \
export DYNACONF_PATHS__CKPT=./data/middle/checkpoints \
./entrypoint.sh

在线
uv run run_algo.py submit-execution -a diagfusion -d ts0-ts-admin-route-service-time-snx7nz --env "CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt" --env "DYNACONF_PATHS__METADATA=./data/middle/metadata" --env "DYNACONF_PATHS__CKPT=./data/middle/checkpoints"
```

# train
```sh
sudo juicefs mount redis://10.10.10.119:6379/1 /mnt/jfs -d --cache-size=1024
export CHECKPOINT_PATH=./data/middle/checkpoints/service_model.pt 
export DYNACONF_PATHS__METADATA=./data/middle/metadata 
export DYNACONF_PATHS__CKPT=./data/middle/checkpoints 
export RCABENCH_BASE_URL=http://127.0.0.1:8082
export RCABENCH_USERNAME=admin
export RCABENCH_PASSWORD=admin123
sudo -E .venv/bin/python  client.py --config gaia_config2.yaml
sudo cp data/rcabench/demo/demo2/dgl/stratification_10/9/topology.pkl data/middle/metadata/ -f
sudo cp data/rcabench/demo/demo2/anomalies/service_instance_mapping.json data/middle/metadata/ -f
sudo rm -rf data/rcabench/demo/demo2
#git restore src/config
docker build -t 10.10.10.240/library/rca-algo-diagfusion:study .
sudo -E .venv/bin/python run.py batch-test --label train1


```
