import json
import os
import argparse
from pathlib import Path
from process_data import process_data
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger, setup_logger
from dataset.dataset_log import derive_filename



def main():
    # 定义目标服务列表
    target_services = [
        'ts-ui-dashboard',
        'ts-travel-plan-service', 
        'ts-route-plan-service',
        'ts-travel2-service',
        'ts-seat-service',
        'ts-order-service'
    ]

    cases = pd.read_parquet(
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_with_issues/index.parquet"
    )
    logger.info(cases.columns)
    
    # 存储符合条件的数据包路径
    data_paths = []
    count = 0
    
    # 遍历所有数据包
    for datapack in cases["datapack"]:
        data_path = Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/{datapack}")
        
        # 获取相关文件路径
        fs = derive_filename(data_path)
        
        # 检查injection文件是否存在
        if "injection" not in fs or not os.path.exists(fs["injection"]):
            logger.warning(f"Injection file not found for {data_path.name}, skipping...")
            continue

        try:
            # 读取injection文件
            with open(fs["injection"], 'r') as f:
                injection = json.load(f)
            
            # 没找到 groundtruth 字段，跳过
            if "ground_truth" not in injection:
                continue
                
            # 获取故障注入的服务名
            service = injection["ground_truth"]["service"][1] if len(injection["ground_truth"]["service"]) > 1 else injection["ground_truth"]["service"][0]
            
            # 检查服务是否在目标列表中
            if service in target_services:
                data_paths.append(data_path)
                logger.info(f"{count}===Found valid case: {data_path.name} with service: {service}")
                count += 1
                
        except Exception as e:
            logger.error(f"Error processing {data_path.name}: {str(e)}")
            continue

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
