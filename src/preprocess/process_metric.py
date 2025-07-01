import os
import json
import pandas as pd
from dataset.k_sigma import Ksigma
from pathlib import Path


def process_parquet_files(data_paths: list[Path]):
    # 初始化 K-sigma 检测器
    detector = Ksigma()
    
    # 存储所有结果,按case_id分组
    case_dict = {}
    
    # 遍历数据包路径
    for case_id, data_pack in enumerate(data_paths, start=0):
        # 获取相关文件路径
        fs = derive_filename(data_pack)
        
        # 检查是否存在 abnormal_metrics.parquet 文件
        parquet_file = fs["abnormal_metrics"]
        if not os.path.exists(parquet_file):
            continue
            
        # 从数据包路径解析instance-id
        parts = data_pack.name.split('-')
        # 获取服务名和实例ID
        service_name = '-'.join(parts[1:4])  # 获取服务名部分
        instance_suffix = parts[-1]  # 获取实例ID后缀
        instance_id = f"{service_name}-{instance_suffix}"  # 组合成完整的instance-id
        
        try:
            # 读取 parquet 文件
            df = pd.read_parquet(parquet_file)
            
            # 确保数据框包含必要的列
            if 'time' not in df.columns or 'value' not in df.columns:
                print(f"跳过 {data_pack.name}: 缺少必要的列")
                continue

            metric_name = df['metric'].iloc[0]  # 从数据中获取metric字段的值
            # 对数据进行排序
            df = df.sort_values('time')
            
            # 获取时间范围
            start_ts = df['time'].min()
            end_ts = df['time'].max()
            
            # 使用 K-sigma 进行检测
            is_anomalous, anomaly_ts, anomaly_score = detector.detection(
                data=df,
                column='value',
                start_ts=start_ts,
                end_ts=end_ts
            )
            
            # 如果检测到异常,添加到对应case_id的列表中
            if is_anomalous:
                # 如果这个case_id还没有对应的列表,创建一个
                if str(case_id) not in case_dict:
                    case_dict[str(case_id)] = []
                
                # 添加异常数据
                case_dict[str(case_id)].append([
                    int(anomaly_ts),
                    instance_id,
                    metric_name,
                    float(anomaly_score)
                ])
                print(f"已处理完成 case {case_id}")
                
        except Exception as e:
            print(f"处理 {data_pack.name} 时出错: {str(e)}")
            continue
    
    # 设置输出路径并确保目录存在
    output_file = Path('../data/rcabench/demo/demo2/anomalies/demo_metric.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(case_dict, f, indent=4)
    
    print(f"处理完成，结果已保存到: {output_file}")
    print(f"共处理了 {len(case_dict)} 个case")
    
    return case_dict

def derive_filename(data_pack: Path) -> dict:
    """
    从数据包路径派生相关文件路径
    
    Args:
        data_pack: 数据包路径
        
    Returns:
        包含相关文件路径的字典
    """
    base_name = data_pack.stem
    # 创建子目录路径
    sub_dir = data_pack.parent / base_name
    return {
        "abnormal_metrics": sub_dir / "abnormal_metrics.parquet",
    }