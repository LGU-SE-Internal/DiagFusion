import json
import os
import pandas as pd
from typing import Optional
from process_metric import process_parquet_files
from dataset.dataset_log import derive_filename, preprocess_logs
from dataset.dataset_trace import save_trace_data
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.config import update_config_nodes


def preprocess_injection(
        data_paths: list[Path],
        output_path: str = "../data/rcabench/demo/demo2/gt.csv",
    ) -> pd.DataFrame:
        """从injection文件中提取信息生成ground truth数据

        Args:
            data_paths: 数据文件路径列表
            output_path: 输出CSV文件路径

        Returns:
            包含ground truth信息的DataFrame
        """
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        records = []
        for idx, data_pack in enumerate(data_paths):
            # 获取相关文件路径
            fs = derive_filename(data_pack)
            
            # 检查injection文件是否存在
            if "injection" not in fs or not os.path.exists(fs["injection"]):
                raise FileNotFoundError(f"Injection file not found for {data_pack.name}")

            # 读取injection文件
            with open(fs["injection"], 'r') as f:
                injection = json.load(f)
            
            # 从injection_name提取service和instance信息
            name_parts = injection["injection_name"].split("-")
            if len(name_parts) >= 3:
                service = "-".join(name_parts[1:4])  # 提取service名称
                instance = f"{service}-{name_parts[-1]}"  # service名称 + 最后的id
            else:
                print(f"Warning: Unexpected injection_name format: {injection['injection_name']}")
                continue

            # 提取时间信息
            start_time = pd.to_datetime(injection["start_time"])
            end_time = pd.to_datetime(injection["end_time"])
            
            # 构建记录
            record = {
                "index": idx,
                "datetime": start_time.date(),
                "service": service,
                "instance": instance,
                "anomaly_type": str(injection["fault_type"]),
                "st_time": start_time,
                "ed_time": end_time,
                "duration": injection["pre_duration"]
            }
            records.append(record)

        # 创建DataFrame并保存
        df = pd.DataFrame(records)
        
        # 添加data_type列,前一半为train,后一半为test
        n = len(df)
        train_size = n // 2
        df['data_type'] = ['train'] * train_size + ['test'] * (n - train_size)
        
        df.to_csv(output_path, index=False)
        print(f"Ground truth数据已保存到: {output_path}")
        
        return df


def process_data(
    data_paths: list[Path],
    config: Optional[dict] = None,
    cache_dir: str = "./cache",
) -> tuple[dict, dict, pd.DataFrame, dict]:
    """
    处理多模态数据,包括trace、日志、注入信息和指标数据

    Args:
        data_paths: 数据文件路径列表
        config: 配置字典,可选
        cache_dir: 缓存目录路径

    Returns:
        tuple包含:
        - trace_dict: trace数据字典
        - processed_logs: 处理后的日志数据
        - injection_df: 注入信息DataFrame  
        - metric_dict: 指标数据字典
    """

    # 预处理日志数据
    processed_logs = preprocess_logs(data_paths, cache_dir)

    # 处理注入信息
    injection_df = preprocess_injection(data_paths)

    # 处理指标数据
    metric_dict = process_parquet_files(data_paths)

    # 处理trace数据
    trace_dict = save_trace_data(data_paths)

    update_config_nodes()

    return trace_dict, processed_logs, injection_df, metric_dict
        

    
    

        
        