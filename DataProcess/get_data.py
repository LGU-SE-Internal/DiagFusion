from pathlib import Path
from preprocess import process_data
import pandas as pd
# Set current working directory
import os
os.chdir('/home/nn/workspace/DiagFusion/DataProcess')


def main():
    cases = pd.read_parquet(
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet"
    )
    print(cases.columns)
    top_10 = cases["datapack"].head(10).tolist()

    data_paths = [
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
