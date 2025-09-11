from dataset.dataset import derive_filename
from pathlib import Path
import pandas as pd
import sys
import os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到Python路径
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, root_dir)
from src.utils.logger import logger


def _build_invoke_links_for_root(df):
    df = df.copy()
    span_to_service = dict(zip(df["span_id"], df["service_name"]))
    df["parent_service"] = df["parent_span_id"].map(span_to_service)
    df["invoke_link"] = (
        df["parent_service"].fillna("ROOT") + "_" + df["service_name"]
    )
    # 将空的 parent_service 填充为 ROOT
    df["parent_service"] = df["parent_service"].fillna("ROOT")
    return df


def find_call_chains_from_root(paths: list[Path]):
    """
    从调用链数据中找出所有以ROOT开头的完整调用链
    
    参数:
    df - 包含调用关系的DataFrame，必须包含parent_service和service_name列
    
    返回:
    list - 每个元素是一个列表，表示从ROOT开始的完整调用链路径
    """
    # 获取前10个data_pack
    data_packs = paths[:100]
    
    # 用于存储所有调用链的结果
    all_chains = []
    call_graph = {}
    all_services = set()  # 存储所有服务

    case_num = 0
    
    # 处理每个data_pack
    for data_pack in data_packs:
        logger.info(f"Processing data pack: {case_num}")
        case_num += 1
        # 获取文件路径
        fs = derive_filename(data_pack)
        
        # 读取异常追踪数据
        abnormal_trace_df = pd.read_parquet(fs["abnormal_trace"])
            
        # 构建调用关系
        df = _build_invoke_links_for_root(abnormal_trace_df)
        
        # 确保df已经处理过，包含parent_service列
        if "parent_service" not in df.columns or "service_name" not in df.columns:
            raise ValueError("DataFrame必须包含parent_service和service_name列")
        
        # 构建服务调用关系图
        for _, row in df.iterrows():
            parent = row["parent_service"]
            service = row["service_name"]
            
            # 添加到所有服务集合
            all_services.add(service)
            if parent != "ROOT":
                all_services.add(parent)
            
            # 跳过自环(parent和service相同的情况)
            if parent == service:
                continue
                
            if parent not in call_graph:
                call_graph[parent] = set()
            call_graph[parent].add(service)
    
    # 找出直接由ROOT调用的服务
    root_services = []
    if "ROOT" in call_graph:
        root_services = list(call_graph["ROOT"])
    
    # 递归查找每个服务的调用链
    def dfs_call_chain(current_service, current_path, visited):
        current_path.append(current_service)
        visited.add(current_service)
        
        # 如果当前服务没有下游调用，则这是一条完整的调用链
        if current_service not in call_graph or not call_graph[current_service]:
            all_chains.append(current_path.copy())
        else:
            # 对每个下游服务，继续递归
            for next_service in call_graph[current_service]:
                # 避免循环调用
                if next_service not in visited:
                    dfs_call_chain(next_service, current_path.copy(), visited.copy())
    
    # 对每个ROOT服务，开始DFS查找调用链
    for service in root_services:
        dfs_call_chain(service, ["ROOT"], set())
    
    # 转换为更易读的格式
    readable_chains = []
    for chain in all_chains:
        readable_chains.append(" -> ".join(chain))
    
    # 将调用链保存到文件中
    output_path = Path("/home/nn/workspace/DiagFusion/src/preprocess/call_chains.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        # 写入所有服务列表
        f.write("所有服务列表:\n")
        f.write(f"总服务数量: {len(all_services)}\n")
        for service in sorted(all_services):
            f.write(f"- {service}\n")
        f.write("-" * 50 + "\n\n")
        
        # 写入根服务列表
        f.write("根服务列表(直接被ROOT调用):\n")
        f.write(f"根服务数量: {len(root_services)}\n")
        for service in sorted(root_services):
            f.write(f"- {service}\n")
        f.write("-" * 50 + "\n\n")
        
        # 写入调用链
        f.write(f"总调用链数量: {len(readable_chains)}\n")
        f.write("-" * 50 + "\n")
        for i, chain in enumerate(readable_chains, 1):
            f.write(f"调用链 {i}:\n{chain}\n")
            f.write("-" * 50 + "\n")
    
    logger.success(f"调用链数据已保存到: {output_path}")
    return readable_chains



def main():
    cases = pd.read_parquet(
        #"/mnt/jfs/rcabench-platform-v2/meta/rcabench_filtered/index.parquet"
        "/mnt/jfs/rcabench-platform-v2/meta/rcabench_with_issues/index.parquet"
    )
    logger.info(cases.columns)
    top_10 = cases["datapack"].head(100).tolist()

    data_paths = [
        #Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/{i}")
        Path(f"/mnt/jfs/rcabench-platform-v2/data/rcabench_with_issues/{i}")
        for i in top_10
    ]

    find_call_chains_from_root(data_paths)


if __name__ == "__main__":
    main()