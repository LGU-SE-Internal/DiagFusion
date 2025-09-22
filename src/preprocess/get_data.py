import os
import sys
from pathlib import Path

import pandas as pd

# Add RCABench imports
from rcabench.openapi import DatasetsApi
from rcabench_platform.v2.clients.rcabench_ import RCABenchClient
from rcabench_platform.v2.datasets.rcabench import valid
from rcabench_platform.v2.logging import timeit
from rcabench_platform.v2.sources.convert import convert_datapack
from rcabench_platform.v2.sources.rcabench import RcabenchDatapackLoader

from src.preprocess.process_data import process_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger


@timeit()
def get_dataset(id: int):
    with RCABenchClient() as client:
        api = DatasetsApi(client)
        resp = api.api_v2_datasets_id_get(id=id, include_injections=True)
        assert resp.data is not None

    assert resp.data.injections is not None
    return resp.data.injections


def main(dataset_id: int = None):
    if dataset_id is None:
        # Fallback to old behavior for backward compatibility
        cases = pd.read_parquet(
            "/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet"
        )
        top_10 = cases["datapack"].head(100).tolist()
        data_paths = [
            Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/{i}")
            for i in top_10
        ]
    else:
        # New behavior: fetch data using RCABench API
        datapacks_injections = get_dataset(dataset_id)
        data_paths = []

        for datapack in datapacks_injections:
            injection_name = datapack.injection_name
            
            # Skip cases containing ui-dashboard
            if "ui-dashboard" in injection_name:
                logger.info(f"Skipping datapack containing ui-dashboard: {injection_name}")
                continue
                
            input_path = Path("data") / "rcabench_dataset" / injection_name
            converted_input_path = input_path / "converted"

            # Validate and convert data
            _, _valid = valid(input_path)
            if not _valid:
                logger.warning(f"Invalid datapack: {injection_name}, skipping")
                continue

            convert_datapack(
                loader=RcabenchDatapackLoader(
                    src_folder=input_path, datapack=injection_name
                ),
                dst_folder=converted_input_path,
                skip_finished=True,
            )

            data_paths.append(converted_input_path)

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
