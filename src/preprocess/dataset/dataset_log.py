import functools
import json
import logging
import os
from pathlib import Path
from typing import Optional
from .utils import CacheManager
import numpy as np
import pandas as pd
from .dataset import RCABenchDataset
from src.utils.logger import logger
import polars as pl
from typing import List
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from .fmap import fmap_processpool


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
        "abnormal_logs": sub_dir / "abnormal_logs.parquet",
        "injection": sub_dir / "injection.json",
    }


def load_injection_data(path: str) -> tuple[str, str]:
    return "mock_fault_type", "mock_target_service"


class DrainProcessor:
    def __init__(self, conf: str, save_path: str, cache_dir: str = "./cache/drain"):
        persistence = FilePersistence(save_path)
        miner_config = TemplateMinerConfig()
        miner_config.load(conf)
        self._template_miner = TemplateMiner(persistence, config=miner_config)
        # self._cache_manager = CacheManager[str](
        #     Path(cache_dir) / "sentence_templates.pkl"
        # )

    def process(self, sentence: str) -> str:
        line = str(sentence).strip()
        if not line:
            return ""

        # cached_result = self._cache_manager.get(line)
        # if cached_result is not None:
        #     return cached_result

        template = self._extract_template(line)
        # self._cache_manager.set(line, template)

        # self.save_cache()
        return template

    # def process_batch(self, sentences: List[str]) -> List[str]:
    #     if not sentences:
    #         return []

    #     results = []
    #     cache_hits = 0
    #     new_templates = {}

    #     # 预处理：去重和清理
    #     unique_sentences = {}
    #     for i, sentence in enumerate(sentences):
    #         line = str(sentence).strip()
    #         if not line:
    #             results.append("")
    #             continue

    #         if line not in unique_sentences:
    #             unique_sentences[line] = []
    #         unique_sentences[line].append(i)

    #     # 初始化结果数组
    #     results = [""] * len(sentences)

    #     # 批量查找缓存
    #     for line, indices in unique_sentences.items():
    #         cached_result = self._cache_manager.get(line)
    #         if cached_result is not None:
    #             cache_hits += len(indices)
    #             for idx in indices:
    #                 results[idx] = cached_result
    #         else:
    #             # 处理新的模板
    #             template = self._extract_template(line)
    #             new_templates[line] = template
    #             for idx in indices:
    #                 results[idx] = template

    #     # 批量更新缓存
    #     if new_templates:
    #         for line, template in new_templates.items():
    #             self._cache_manager.set(line, template)

    #     if new_templates:
    #         self.save_cache()

    #     return results

    def _extract_template(self, line: str) -> str:
        result = self._template_miner.add_log_message(line)
        template = result.get("template_mined")
        if template is None:
            logger.warning(f"Failed to extract template for: {line}")
            return ""
        return template

    # def save_cache(self):
    #     self._cache_manager.save()


def process_datapack(data_pack: Path) -> list:
    drain = DrainProcessor(conf="/home/nn/workspace/DiagFusion/drain.ini", save_path="cache/drain/temp")
        
    # 获取相关文件路径
    fs = derive_filename(data_pack)

    # 检查必要文件是否存在
    if "abnormal_logs" not in fs or not os.path.exists(fs["abnormal_logs"]):
        raise FileNotFoundError(f"Abnormal log file not found for {data_pack.name}")

    # 读取并预处理数据
    lf = pl.scan_parquet(fs["abnormal_logs"])
    lf = lf.filter(pl.col("service_name") != "ts-ui-dashboard").sort("time")

    # 时间转换为毫秒级时间戳
    lf = lf.with_columns([
        pl.col("time").dt.timestamp(time_unit="ms").alias("time")
    ])

    # 提取日志模板并计算模板ID
    df = lf.collect()

    # 获取当前datapack的唯一消息
    unique_messages = df["message"].unique().to_list()

    # 对当前datapack的唯一消息进行drain处理
    message_to_template = {}
    message_to_template_id = {}
    #for msg in enumerate(unique_messages):
    for msg in unique_messages:
        template = drain.process(msg)
        template_id = generate_template_id(template)
        message_to_template[msg] = template
        message_to_template_id[msg] = template_id

    # 映射回当前datapack的数据
    df = df.with_columns([
        pl.col("message").map_elements(lambda x: message_to_template[x], return_dtype=pl.Utf8).alias("template"),
        pl.col("message").map_elements(lambda x: message_to_template_id[x], return_dtype=pl.Utf8).alias("template_id")
    ])

    # 保存缓存
    # drain.save_cache()

    # 将当前data_pack的日志转换为[time, service_name, template_id]格式的列表
    log_sequence = df.select(["time", "service_name", "template_id"]).to_numpy().tolist()

    return log_sequence


def preprocess_logs(
    data_paths: list[Path],
    cache_dir: str = "./cache",
    max_workers: Optional[int] = None,
) -> list[list]:
    """预处理日志数据的主函数

    Args:
        data_paths: 数据文件路径列表
        cache_dir: 缓存目录
        max_workers: 最大并行工作进程数

    Returns:
        每个datapack对应的日志序列列表
    """

    # drain = DrainProcessor(conf="/home/nn/workspace/DiagFusion/drain.ini", save_path="cache/drain/temp")

    # 存储每个data_pack的日志序列
    all_log_sequences = []
    
    # # 为每个datapack单独处理，串行处理
    # for data_pack in data_paths:
    #     log_sequence = process_datapack(data_pack, drain)
        
    #     all_log_sequences.append(log_sequence)
        
    #     logger.info(f"数据包 {data_pack.name} 处理完成，日志条数: {len(log_sequence)}")
    #     if log_sequence:
    #         logger.info(f"示例日志: {log_sequence[0]}")

    # 并行处理

    n_workers = 8
    
    # 创建带索引的任务，以便后续排序
    indexed_tasks = [
        (i, functools.partial(
            process_datapack,
            data_pack,
        ))
        for i, data_pack in enumerate(data_paths)
    ]
    
    # 存储带索引的结果
    indexed_results = []

    for i in range(0, len(indexed_tasks), n_workers):
        batch_indexed_tasks = indexed_tasks[i : i + n_workers]
        
        # 提取任务函数进行并行处理
        batch_tasks = [task for _, task in batch_indexed_tasks]
        batch_indices = [idx for idx, _ in batch_indexed_tasks]

        batch_results = fmap_processpool(
            batch_tasks, parallel=n_workers, ignore_exceptions=False
        )

        # 将结果与索引配对
        for idx, result in zip(batch_indices, batch_results):
            indexed_results.append((idx, result))

    # 按原始索引排序，确保顺序与 data_paths 一致
    indexed_results.sort(key=lambda x: x[0])
    
    # 提取排序后的结果
    all_log_sequences = [result for _, result in indexed_results]

    # # 保存缓存
    # drain.save_cache()

    # 保存为npy文件，现在包含多个独立的列表
    save_path = Path(
        "./data/rcabench/demo/demo2/anomalies/stratification_logs.npy"
    ).resolve()
    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), np.array(all_log_sequences, dtype=object))

    return all_log_sequences


def preprocess_logs_inference(
    data_path: Path,
    output_path: Path,
    cache_dir: str = "./cache",
) -> list[pd.DataFrame]:
    """预处理日志数据的主函数

    Args:
        data_paths: 数据文件路径列表
        cache_dir: 缓存目录
        max_workers: 最大并行工作进程数

    Returns:
        处理后的DataFrame列表
    """
    # 初始化处理器
    drain = DrainProcessor(conf="/home/nn/workspace/DiagFusion/drain.ini", save_path="cache/drain/temp")

    # 存储每个data_pack的日志序列
    log_sequences = []

    lf = pl.scan_parquet(os.path.join(data_path, "abnormal_logs.parquet"))
    lf = lf.filter(pl.col("service_name") != "ts-ui-dashboard").sort("time")

    # 时间转换为毫秒级时间戳
    lf = lf.with_columns([
        pl.col("time").dt.timestamp(time_unit="ms").alias("time")
    ])

    df = lf.collect()

    # 获取唯一的消息
    unique_messages = df["message"].unique().to_list()

    # 对唯一消息进行drain处理
    message_to_template = {}
    message_to_template_id = {}
    for msg in unique_messages:
        template = drain.process(msg)
        template_id = generate_template_id(template)
        message_to_template[msg] = template
        message_to_template_id[msg] = template_id

    # 映射回原始数据
    df = df.with_columns([
        pl.col("message").map_elements(lambda x: message_to_template[x], return_dtype=pl.Utf8).alias("template"),
        pl.col("message").map_elements(lambda x: message_to_template_id[x], return_dtype=pl.Utf8).alias("template_id")
    ])

    # 将当前data_pack的日志转换为[time, service_name, template_id]格式的列表
    log_sequences = df.select(["time", "service_name", "template_id"]).to_numpy().tolist()

    # 保存缓存
    drain.save_cache()

    save_path = os.path.join(output_path, "stratification_logs.npy")
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(str(save_path), np.array(log_sequences, dtype=object))

    logger.success(f"已保存日志序列，形状: {len(log_sequences)}")
    logger.info(f"第一个序列示例: {log_sequences[0][:3]}")

    return log_sequences


# 保证了对于相同的模板一定会生成相同长度的ID
def generate_template_id(template: str) -> str:
    """生成固定长度的模板ID
    Args:
        template: 日志模板字符串
    Returns:
        8位的十六进制哈希值
    """
    import hashlib
    # 使用MD5生成确定性哈希，确保相同模板总是产生相同ID
    hash_value = hashlib.md5(template.encode('utf-8')).hexdigest()
    # 取前8位作为模板ID
    return hash_value[:8]