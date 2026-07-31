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


class FilePersistence:
    def __init__(self, save_path):
        pass


class TemplateMinerConfig:
    def load(self, path):
        pass


class TemplateMiner:
    def __init__(self, persistence, config):
        pass

    def add_log_message(self, line: str) -> dict:
        template = " ".join([word for word in line.split() if not word.isdigit()])
        return {"template_mined": template or "default_template"}


class DrainProcesser:
    def __init__(self, conf: str, save_path: str, cache_dir: str = "./cache/drain"):
        persistence = FilePersistence(save_path)
        miner_config = TemplateMinerConfig()
        miner_config.load(conf)
        self._template_miner = TemplateMiner(persistence, config=miner_config)
        self._cache_manager = CacheManager[str](
            Path(cache_dir) / "sentence_templates.pkl"
        )

    def __call__(self, sentence: str) -> str:
        """Processes a log message to extract its template, using a cache."""
        line = str(sentence).strip()
        if not line:
            return ""

        return self._cache_manager.get_or_compute(
            line, lambda: self._process_line(line)
        )

    def _process_line(self, line: str) -> str:
        """Internal logic for processing a single log line with Drain."""
        result = self._template_miner.add_log_message(line)
        template = result.get("template_mined")
        if template is None:
            logging.warning(
                f"Failed to find 'template_mined' for line: {line}. Result: {result}"
            )
            return ""  # Return a default or empty template
        return template

    def save_cache(self):
        self._cache_manager.save()


class LogDataset(RCABenchDataset):
    def __init__(
        self,
        paths: list[Path],
        cache_dir: str = "./cache",
        max_workers: Optional[int] = None,
    ):
        # Initialize drain and encoder before calling super()
        self._drain = DrainProcesser(
            "dataset/drain3/drain.ini", "data/gaia/drain.bin", f"{cache_dir}/drain"
        )

        super().__init__(
            paths,
            transform=self._transform_log,
            cache_dir=cache_dir,
            cache_name="dataset_log",
            use_dataset_cache=True,
            max_workers=max_workers,
        )

    def _save_all_caches(self):
        """A single place to save all underlying caches."""
        super()._save_all_caches()
        self._drain.save_cache()
        self._encoder.save_cache()

    def _transform_log(self, data_pack: Path) -> tuple:
        """The core transformation logic for a single log data pack."""
        fs = derive_filename(data_pack)
        if "abnormal_log" not in fs or not os.path.exists(fs["abnormal_log"]):
            raise FileNotFoundError(f"Abnormal log file not found for {data_pack.name}")
        if "injection" not in fs or not os.path.exists(fs["injection"]):
            raise FileNotFoundError(f"Injection file not found for {data_pack.name}")

        df = pd.read_parquet(fs["abnormal_log"])
        df = df[df["service_name"] != "ts-ui-dashboard"].sort_values(by="time")
        fault_type, target_service = load_injection_data(str(fs["injection"]))

        df["time_bucket"] = pd.to_datetime(df["time"], unit="s").dt.floor("min")

        # Process logs minute by minute
        seqs, cnt_of_log = [], {}
        for minute, group in df.groupby("time_bucket"):
            templates = [self._drain(log) for log in group["message"]]
            seqs.append(list(set(templates)))
            for template in templates:
                cnt_of_log.setdefault(template, [0] * len(df["time_bucket"].unique()))[
                    len(seqs) - 1
                ] += 1

        # Calculate weights for each log template based on frequency change
        wei_of_log = {}
        total_gap = 1e-5
        for template, counts in cnt_of_log.items():
            log_counts = np.log(np.array(counts) + 1e-5)
            change = np.abs(np.diff(np.insert(log_counts, 0, 0)))
            gap = change.max() - change.mean()
            wei_of_log[template] = gap
            total_gap += gap

        # Encode all unique templates
        all_templates = list(set(t for seq in seqs for t in seq))
        template_embeddings = self._encoder.batch_encode(all_templates, batch_size=64)
        template_to_embedding = {
            t: np.array(e) for t, e in zip(all_templates, template_embeddings)
        }

        # Create weighted sequence representations
        final_sequence = []
        for seq in seqs:
            repr_vec = np.zeros(self._encoder._model.config.hidden_size)
            for template in seq:
                if template in template_to_embedding:
                    repr_vec += (
                        wei_of_log[template] / total_gap
                    ) * template_to_embedding[template]
            final_sequence.append(repr_vec.tolist())

        labels = {"fault_type": fault_type, "target_service": target_service}
        return final_sequence, labels


def preprocess_logs(
    data_paths: list[Path],
    cache_dir: str = "./cache",
    max_workers: Optional[int] = None,
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
    drain = DrainProcesser(
        "dataset/drain3/drain.ini", "data/gaia/drain.bin", f"{cache_dir}/drain"
    )

    # 存储每个data_pack的日志序列
    log_sequences = []
    lfs = []
    for data_pack in data_paths:
        # 获取相关文件路径
        fs = derive_filename(data_pack)

        # 检查必要文件是否存在
        if "abnormal_logs" not in fs or not os.path.exists(fs["abnormal_logs"]):
            raise FileNotFoundError(f"Abnormal log file not found for {data_pack.name}")

        # 读取并预处理数据
        lf = pl.scan_parquet(fs["abnormal_logs"])
        lfs.append(lf)
    lf = pl.concat(lfs)
    lf = lf.filter(pl.col("service_name") != "ts-ui-dashboard").sort("time")

    # 时间转换为毫秒级时间戳
    lf = lf.with_columns([
        pl.col("time").dt.timestamp(time_unit="ms").alias("time")
    ])

    # 提取日志模板并计算模板ID
    df = lf.collect()

    # 获取唯一的消息
    unique_messages = df["message"].unique().to_list()

    # 对唯一消息进行drain处理
    message_to_template = {}
    message_to_template_id = {}
    for msg in unique_messages:
        template = drain(msg)
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

    # 保存为npy文件
    save_path = Path(
        "./data/rcabench/demo/demo2/anomalies/stratification_logs.npy"
    ).resolve()
    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), np.array(log_sequences, dtype=object))

    logger.success(f"已保存日志序列，形状: {len(log_sequences)}")
    logger.info(f"第一个序列示例: {log_sequences[0][:3]}")

    return log_sequences


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
    drain = DrainProcesser(
        "dataset/drain3/drain.ini", "data/gaia/drain.bin", f"{cache_dir}/drain"
    )

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
        template = drain(msg)
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
    if log_sequences:
        logger.info(f"第一个序列示例: {log_sequences[0][:3]}")
    else:
        logger.info("当前数据包没有可用日志序列")

    return log_sequences


# 保证了对于相同的模板一定会生成相同长度的ID
def generate_template_id(template: str) -> str:
    """生成固定长度的模板ID
    Args:
        template: 日志模板字符串
    Returns:
        8位的十六进制哈希值
    """
    hash_value = abs(hash(template))
    hex_str = hex(hash_value)[2:]  # 去掉'0x'前缀
    # 如果长度不足8位，在前面补0；如果超过8位，取后8位
    return hex_str.zfill(8)[-8:]
