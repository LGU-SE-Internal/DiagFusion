from pathlib import Path
from process_data import process_data
import pandas as pd
# Set current working directory



def main():
    cases = pd.read_parquet(
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_with_issues/index.parquet"
    )
    print(cases.columns)
    top_10 = cases["datapack"].head(100).tolist()

    data_paths = [
        Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/{i}")
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
