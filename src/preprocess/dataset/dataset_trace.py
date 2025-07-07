import pickle
from .dataset import RCABenchDataset, derive_filename
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from .utils import load_injection_data, CacheManager
import hashlib
import json
from dataset.k_sigma import Ksigma
from src.utils.logger import logger
import gc


# 使用示例
def save_trace_data(data_paths, output_path="../data/rcabench/demo/demo2/anomalies/demo_trace.json"):
    """处理并保存trace数据为JSON文件"""
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  
    # 预处理数据
    trace_dict = preprocess_trace_data(data_paths, cache_dir="./cache")
    
    # 保存为JSON文件
    with open(output_path, 'w') as f:
        json.dump(trace_dict, f, indent=4)
    
    logger.success(f"Trace数据已保存到: {output_path}")
    logger.info(f"共处理了 {len(trace_dict)} 个case")
    return trace_dict


def preprocess_trace_data(paths: list[Path], cache_dir: str = "./cache") -> dict:
    processed_dict = {}
    all_traces = []
    batch_size = 5  # 每批处理5个数据包
    
    # 分批处理数据包
    for batch_idx in range(0, len(paths), batch_size):
        batch_paths = paths[batch_idx:batch_idx + batch_size]
        batch_traces = []
        
        # 处理当前批次的数据包
        for case_id, data_pack in enumerate(batch_paths, start=batch_idx):
            try:
                fs = derive_filename(data_pack)
                
                # 读取injection文件
                with open(fs["injection"], 'r') as f:
                    injection = json.load(f)
            
                # 没找到 groundtruth 字段，跳过
                if "ground_truth" not in injection:
                    continue
                
                
                abnormal_trace_df = pd.read_parquet(fs["abnormal_trace"])
                
                # 转换时间戳
                abnormal_trace_df["timestamp"] = abnormal_trace_df["time"].apply(
                    lambda x: int(x.timestamp() * 1000)
                    if hasattr(x, "timestamp")
                    else int(x / 1000)
                )

                # 构建调用关系
                abnormal_trace_df = _build_invoke_links(abnormal_trace_df)
                
                # 收集trace数据用于生成topology（只保留必要的列以节省内存）
                batch_traces.append(abnormal_trace_df[["service_name", "parent_service"]])
                
                # 使用 k_sigma 计算异常分数
                k_sigma = Ksigma()
                scores = []
                for name, group in abnormal_trace_df.groupby(["service_name", "parent_service"]):
                    group = group.sort_values("timestamp")
                    if len(group) > 0:
                        is_anomaly, _, score = k_sigma.detection(
                            data=pd.DataFrame({
                                "time": group["timestamp"].values,
                                "value": group["duration"].values
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
                
                # 只保存结果，不保存整个DataFrame
                if trace_records:
                    processed_dict[str(case_id)] = trace_records
                    logger.info(f"已处理完成 case {case_id}，生成 {len(trace_records)} 条记录")
                
                # 及时释放内存
                del abnormal_trace_df
                del scores
                
            except Exception as e:
                logger.error(f"处理 case {case_id} 时出错: {str(e)}")
                continue
        
        # 处理完一批后，更新all_traces并释放内存
        if batch_traces:
            # 只合并拓扑所需的数据
            concat_df = pd.concat(batch_traces)
            all_traces.append(concat_df)
            del batch_traces
            
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 可选：每处理一批后将中间结果保存到磁盘
        # temp_file = f"temp_processed_dict_batch_{batch_idx}.pkl"
        # with open(temp_file, 'wb') as f:
        #     pickle.dump(processed_dict, f)
        # print(f"已保存批次 {batch_idx} 的中间结果到 {temp_file}")
    
    generate_service_topology(all_traces)
    
        
    return processed_dict


def generate_service_topology(all_traces):
    # 合并所有trace数据并生成topology
    if all_traces:
        all_traces_df = pd.concat(all_traces, ignore_index=True)

        # 处理 groundtruth
        gt_path = "../data/rcabench/demo/demo2/gt.csv"
        gt_df = pd.read_csv(gt_path)
        services = gt_df["service"].unique().tolist()
        # 服务ID到服务名称的映射
        service_id_to_name = {
            idx: service_name
            for idx, service_name in enumerate(services)
        }

        # 服务名称到ID的映射
        service_to_id = {
            service_name: idx
            for idx, service_name in enumerate(services)
        }
        # 创建实例到ID的映射
        instance_to_id = service_to_id

        topology = _generate_topology(all_traces_df, service_to_id)
        
        # 将topology保存到文件
        topology_path = Path("../data/rcabench/demo/demo2/anomalies/trace_topology.json")
        # 确保目录存在
        topology_path.parent.mkdir(parents=True, exist_ok=True)
        with open(topology_path, 'w') as f:
            json.dump({
                "source_nodes": topology[0],
                "target_nodes": topology[1]
            }, f, indent=4)
        logger.success(f"服务拓扑已保存到: {topology_path}")

        service_to_instance(topology, gt_df, service_id_to_name, instance_to_id)



def _build_invoke_links(df):
    df = df.copy()
    span_to_service = dict(zip(df["span_id"], df["service_name"]))
    df["parent_service"] = df["parent_span_id"].map(span_to_service)
    df["invoke_link"] = (
        df["parent_service"].fillna("ROOT") + "_" + df["service_name"]
    )
    df = df[df["parent_service"].notna()]
    return df
    
def _generate_topology(all_traces_df, service_to_id):
    """从trace数据生成拓扑结构
        
    Args:
        all_traces_df: 包含所有trace数据的DataFrame
        service_to_id: 服务名称到ID的映射字典
            
    Returns:
        tuple: (source_nodes, target_nodes) 分别表示边的源节点和目标节点列表
    """
    # 从gt.csv中读取服务列表
    gt_path = "../data/rcabench/demo/demo2/gt.csv"
    gt_df = pd.read_csv(gt_path)
    target_services = gt_df["service"].unique().tolist()
        
    # 获取所有唯一的服务名称并排序
    all_services = sorted(list(set(
        list(all_traces_df["service_name"].unique()) + 
        list(all_traces_df["parent_service"].unique())
    )))
        
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
            # 添加双向边,但跳过自环
            # if source_id != target_id:
            edges.add((source_id, target_id))
            edges.add((target_id, source_id))
        
    # 转换为源节点和目标节点列表
    source_nodes = []
    target_nodes = []
    for source, target in sorted(edges):
        source_nodes.append(source)
        target_nodes.append(target)
    
    # 检查是否缺少节点
    num_services = len(service_to_id)
    existing_nodes = set(source_nodes)
    for i in range(num_services):
        if i not in existing_nodes:
            # 如果缺少节点i，添加一条从i到i的边
            source_nodes.append(i)
            target_nodes.append(i)
            
    return source_nodes, target_nodes


def service_to_instance(topology=None, gt_df=None, service_id_to_name=None, instance_to_id=None):
    """将服务级别的拓扑转换为实例级别的拓扑

    Args:
        topology: tuple, 包含source_nodes和target_nodes的元组

    Returns:
        source_nodes: 源节点ID列表
        target_nodes: 目标节点ID列表
    """

    # 构建service_to_instances映射
    service_to_instances = {}
    for service in gt_df["service"].unique():
        instances = list(set(gt_df[gt_df["service"] == service]["instance"].tolist()))
        service_to_instances[service] = instances

    # 创建服务ID到实例ID的映射
    service_instance_map = {}
    for service_id, service_name in service_id_to_name.items():
        instance_ids = [instance_to_id[inst] for inst in service_to_instances[service_name]]
        service_instance_map[service_id] = instance_ids

    # 保存映射关系
    mapping_path = Path("../data/rcabench/demo/demo2/anomalies/service_instance_mapping.json")
    # 确保目录存在
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump({
            "service_id_to_name": service_id_to_name,
            "service_instance_map": service_instance_map,
            "instance_to_id": instance_to_id
        }, f, indent=4)
    logger.success(f"服务ID与实例ID的映射已保存到: {mapping_path}")

    if topology is None:
        # 如果没有传入topology，则从文件读取
        topology_path = "../data/rcabench/demo/demo2/anomalies/trace_topology.json"
        with open(topology_path, 'r') as f:
            topology_data = json.load(f)
            topology = (topology_data["source_nodes"], topology_data["target_nodes"])

    # 生成实例级别的拓扑
    source_nodes = []
    target_nodes = []

    # 遍历服务级别的每条边
    for src, tgt in zip(topology[0], topology[1]):
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
    instance_topology_path = Path("../data/rcabench/demo/demo2/anomalies/instance_topology.json")
    # 确保目录存在
    instance_topology_path.parent.mkdir(parents=True, exist_ok=True)
    with open(instance_topology_path, 'w') as f:
        json.dump({
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "instance_to_id": instance_to_id
        }, f, indent=4)
    logger.success(f"实例级别的拓扑已保存到: {instance_topology_path}")