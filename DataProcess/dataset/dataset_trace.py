from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from .utils import load_injection_data, CacheManager
import hashlib
import json
import os
os.chdir('/home/nn/workspace/DiagFusion/DataProcess')
from dataset.k_sigma import Ksigma

    

def preprocess_trace_data(paths: list[Path], cache_dir: str = "./cache") -> dict:
    """预处理trace数据并返回指定格式的字典
    
    Returns:
        dict: 格式为 {"case_id": [[timestamp, parent_service, service_name, score], ...]}
    """
    processed_dict = {}
    
    def _build_invoke_links(df):
        df = df.copy()
        span_to_service = dict(zip(df["span_id"], df["service_name"]))
        df["parent_service"] = df["parent_span_id"].map(span_to_service)
        df["invoke_link"] = (
            df["parent_service"].fillna("ROOT") + "_" + df["service_name"]
        )
        df = df[df["parent_service"].notna()]
        return df
    
    def _generate_topology(all_traces_df):
        """从trace数据生成拓扑结构
        
        Args:
            all_traces_df: 包含所有trace数据的DataFrame
            
        Returns:
            tuple: (source_nodes, target_nodes) 分别表示边的源节点和目标节点列表
        """
        # 从gt.csv中读取服务列表
        gt_path = "/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/gt.csv"
        gt_df = pd.read_csv(gt_path)
        target_services = gt_df["service"].unique().tolist()
        
        # 获取所有唯一的服务名称并排序
        all_services = sorted(list(set(
            list(all_traces_df["service_name"].unique()) + 
            list(all_traces_df["parent_service"].unique())
        )))
        
        # 过滤出目标服务
        filtered_services = [s for s in all_services if s in target_services]
        
        # 创建服务名到ID的映射
        service_to_id = {service: idx for idx, service in enumerate(filtered_services)}
        
        # 收集所有的服务调用关系
        edges = set()
        # 过滤出只包含目标服务的调用关系
        filtered_df = all_traces_df[
            (all_traces_df["service_name"].isin(target_services)) & 
            (all_traces_df["parent_service"].isin(target_services))
        ]
        unique_calls = filtered_df[["parent_service", "service_name"]].drop_duplicates()
        
        for _, row in unique_calls.iterrows():
            if pd.notna(row["parent_service"]):  # 确保parent_service不是NA
                source_id = service_to_id[row["parent_service"]]
                target_id = service_to_id[row["service_name"]]
                # 添加双向边
                edges.add((source_id, target_id))
                edges.add((target_id, source_id))
        
        # 转换为源节点和目标节点列表
        source_nodes = []
        target_nodes = []
        for source, target in sorted(edges):
            source_nodes.append(source)
            target_nodes.append(target)
            
        return source_nodes, target_nodes

    # 用于收集所有trace数据的列表
    all_traces = []

    # 处理每个数据包
    for case_id, data_pack in enumerate(paths, start=0):
        try:
            fs = derive_filename(data_pack)
            abnormal_trace_df = pd.read_parquet(fs["abnormal_trace"])
            
            # 转换时间戳
            abnormal_trace_df["timestamp"] = abnormal_trace_df["time"].apply(
                lambda x: int(x.timestamp() * 1000)  # 转换为毫秒级时间戳
                if hasattr(x, "timestamp")
                else int(x / 1000)  # 假设输入是微秒，转换为毫秒
            )

            # 构建调用关系
            abnormal_trace_df = _build_invoke_links(abnormal_trace_df)
            
            # 收集trace数据用于生成topology
            all_traces.append(abnormal_trace_df)
            
            # 使用 k_sigma 计算异常分数
            k_sigma = Ksigma()
            scores = []
            for name, group in abnormal_trace_df.groupby(["service_name", "parent_service"]):
                group = group.sort_values("timestamp")
                if len(group) > 0:
                    # 对每个组使用 k_sigma 检测
                    is_anomaly, _, score = k_sigma.detection(
                        data=pd.DataFrame({
                            "time": group["timestamp"].values,
                            "value": group["duration"].values  # 使用duration字段进行异常检测
                        }),
                        column="value",
                        start_ts=group["timestamp"].min(),
                        end_ts=group["timestamp"].max()
                    )
                    scores.extend([abs(score) if is_anomaly else 0] * len(group))
                else:
                    scores.extend([0] * len(group))
            
            abnormal_trace_df["score"] = scores

            # 提取需要的列并转换为列表格式
            trace_records = abnormal_trace_df[["timestamp", "parent_service", "service_name", "score"]].values.tolist()
            
            # 只有当有记录时才添加到结果字典
            if trace_records:
                processed_dict[str(case_id)] = trace_records
                print(f"已处理完成 case {case_id}")

        except Exception as e:
            print(f"处理 case {case_id} 时出错: {str(e)}")
            continue

    # 合并所有trace数据并生成topology
    if all_traces:
        all_traces_df = pd.concat(all_traces, ignore_index=True)
        topology = _generate_topology(all_traces_df)
        # 将topology保存到文件
        topology_path = "/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/anomalies/trace_topology.json"
        with open(topology_path, 'w') as f:
            json.dump({
                "source_nodes": topology[0],
                "target_nodes": topology[1]
            }, f, indent=4)
        print(f"Topology已保存到: {topology_path}")


    service_to_instance()

    return processed_dict

# 使用示例
def save_trace_data(data_paths, output_path="/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/anomalies/demo_trace.json"):
    """处理并保存trace数据为JSON文件"""
    # 预处理数据
    trace_dict = preprocess_trace_data(data_paths, cache_dir="./cache")
    
    # 保存为JSON文件
    with open(output_path, 'w') as f:
        json.dump(trace_dict, f, indent=4)
    
    print(f"Trace数据已保存到: {output_path}")
    print(f"共处理了 {len(trace_dict)} 个case")
    return trace_dict


def service_to_instance():
    """将服务级别的拓扑转换为实例级别的拓扑

    Args:
        service_name: 服务名称

    Returns:
        source_nodes: 源节点ID列表
        target_nodes: 目标节点ID列表
    """
    # 从gt.csv中读取服务列表并创建映射
    gt_path = "/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/gt.csv"
    gt_df = pd.read_csv(gt_path)
    services = sorted(gt_df["service"].unique().tolist())
    
    # 服务ID到服务名称的映射
    service_id_to_name = {
        idx: service_name
        for idx, service_name in enumerate(services)
    }

    # 构建service_to_instances映射
    service_to_instances = {}
    for service in gt_df["service"].unique():
        instances = gt_df[gt_df["service"] == service]["instance"].tolist()
        service_to_instances[service] = instances

    # 创建实例到ID的映射
    instance_to_id = {}
    for i, instance in enumerate(gt_df['instance']):
        instance_to_id[instance] = i

    # 读取服务级别的拓扑
    topology_path = "/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/anomalies/trace_topology.json"
    with open(topology_path, 'r') as f:
        service_topology = json.load(f)

    # 生成实例级别的拓扑
    source_nodes = []
    target_nodes = []

    # 遍历服务级别的每条边
    for src, tgt in zip(service_topology["source_nodes"], service_topology["target_nodes"]):
        src_service = service_id_to_name[src]
        tgt_service = service_id_to_name[tgt]

        # 获取源服务和目标服务的所有实例
        src_instances = service_to_instances[src_service]
        tgt_instances = service_to_instances[tgt_service]

        # 为每对实例创建边
        for src_instance in src_instances:
            for tgt_instance in tgt_instances:
                source_nodes.append(instance_to_id[src_instance])
                target_nodes.append(instance_to_id[tgt_instance])

    # 保存实例级别的拓扑
    instance_topology_path = "/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/anomalies/instance_topology.json"
    with open(instance_topology_path, 'w') as f:
        json.dump({
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "instance_to_id": instance_to_id
        }, f, indent=4)
    print(f"实例级别的拓扑已保存到: {instance_topology_path}")