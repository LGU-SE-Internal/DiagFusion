import json
import os
import argparse
from pathlib import Path
import pandas as pd
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.preprocess.process_data import process_data_inference
from src.utils.logger import logger, setup_logger
from src.preprocess.dataset.dataset_log import derive_filename


def main():
    data_path = Path("/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/ts0-ts-basic-service-response-delay-7fxg9f")

    config = {
        "max_len": 512,
        "d_model": 768,
        "nhead": 8,
        "d_ff": 2048,
        "layer_num": 6,
        "dropout": 0.1,
    }

    process_data_inference(data_path, config, cache_dir="./cache")


if __name__ == "__main__":
    main()
