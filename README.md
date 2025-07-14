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
python main.py --config gaia_config.yaml
./main.py eval single diagfusion rcabench_filtered ts3-ts-travel-service-response-delay-s4plkv
./main.py eval batch -d rcabench_filtered -a diagfusion --clear
```

```bash
INPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-food-delivery-service-exception-f5xmtg  \
OUTPUT_PATH=/mnt/jfs/rcabench_dataset/ts0-ts-food-delivery-service-exception-f5xmtg \
CHECKPOINT_PATH=/home/nn/workspace/DiagFusion/src/data/rcabench/demo/demo2/dgl/stratification_10/9/service_model.pt \
./entrypoint.sh
```

