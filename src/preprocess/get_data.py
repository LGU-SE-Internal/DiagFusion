import json
import os
import argparse
from pathlib import Path
from src.preprocess.process_data import process_data
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger, setup_logger
from src.preprocess.dataset.dataset_log import derive_filename


def main():
    cases = pd.read_parquet(
        # "/mnt/jfs/rcabench-platform-v2/meta/rcabench_with_issues/index.parquet"
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet"
    )
    top_10 = cases["datapack"].head(10).tolist()
    data_paths = [
        # Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/{i}")
        Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/{i}")
        for i in top_10
    ]

    config = {
        "max_len": 512,
        "d_model": 768,
        "nhead": 8,
        "d_ff": 2048,
        "layer_num": 6,
        "dropout": 0.1,
    }

    process_data(data_paths, config, cache_dir="./cache")


if __name__ == "__main__":
    main()
