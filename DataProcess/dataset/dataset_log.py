import logging
import os
from pathlib import Path
from typing import Optional
from .utils import CacheManager
import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from .dataset import RCABenchDataset

_global_bert_encoder = None


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


class BertEncoder:
    def __init__(self, cache_dir: str = "./cache/bert_encoder"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"BertEncoder using device: {self.device}")

        # Using a shared cache directory for model weights
        model_cache_dir = "./cache/bert_models"
        self._tokenizer = BertTokenizer.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir=model_cache_dir
        )
        self._model = BertModel.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir=model_cache_dir
        )
        self._model.to(self.device).eval()  # type: ignore

        self._cache_manager = CacheManager[list[float]](
            Path(cache_dir) / "sentence_embeddings.pkl"
        )

    @staticmethod
    def get_global_instance(cache_dir: str = "./cache/bert_encoder"):
        global _global_bert_encoder
        if _global_bert_encoder is None:
            _global_bert_encoder = BertEncoder(cache_dir)
        return _global_bert_encoder

    def _get_cache_key(self, sentence: str, no_wordpiece: bool) -> str:
        return f"{sentence}|{no_wordpiece}"

    def encode(self, sentence: str, no_wordpiece: bool = False) -> list[float]:
        """Encodes a single sentence, using the cache if available."""
        key = self._get_cache_key(sentence, no_wordpiece)
        return self._cache_manager.get_or_compute(
            key, lambda: self._encode_single(sentence, no_wordpiece)
        )

    def batch_encode(
        self,
        sentences: list[str],
        no_wordpiece: bool = False,
        batch_size: int = 32,
        save_every: int = 100,
    ) -> list[list[float]]:
        """Encodes a batch of sentences, leveraging the cache and processing only new sentences."""
        results: list[Optional[list[float]]] = [None] * len(sentences)
        uncached_sentences: list[str] = []
        uncached_indices: list[int] = []

        for i, sentence in enumerate(sentences):
            key = self._get_cache_key(sentence, no_wordpiece)
            cached_embedding = self._cache_manager.get(key)
            if cached_embedding is not None:
                results[i] = cached_embedding
            else:
                uncached_sentences.append(sentence)
                uncached_indices.append(i)

        if not uncached_sentences:
            return results  # type: ignore

        new_embeddings = self._batch_encode_internal(
            uncached_sentences, no_wordpiece, batch_size
        )

        for i, (original_index, embedding) in enumerate(
            zip(uncached_indices, new_embeddings)
        ):
            results[original_index] = embedding
            key = self._get_cache_key(uncached_sentences[i], no_wordpiece)
            self._cache_manager.set(key, embedding)
            if (i + 1) % save_every == 0:
                self.save_cache()

        self.save_cache()
        return results  # type: ignore

    def _encode_single(self, sentence: str, no_wordpiece: bool = False) -> list[float]:
        """Internal logic for encoding a single sentence."""
        return self._batch_encode_internal([sentence], no_wordpiece, batch_size=1)[0]

    def _batch_encode_internal(
        self, sentences: list[str], no_wordpiece: bool, batch_size: int
    ) -> list[list[float]]:
        """Internal logic for encoding a batch of sentences on the ML device."""
        if no_wordpiece:
            sentences = [
                " ".join(w for w in s.split() if w in self._tokenizer.vocab)
                for s in sentences
            ]

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
        return all_embeddings

    def save_cache(self):
        self._cache_manager.save()

    def __call__(self, sentence: str, no_wordpiece: bool = False) -> list[float]:
        return self.encode(sentence, no_wordpiece)


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
        self._encoder = BertEncoder.get_global_instance(f"{cache_dir}/bert_encoder")

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
        "dataset/drain3/drain.ini",
        "data/gaia/drain.bin", 
        f"{cache_dir}/drain"
    )
    
    # 存储每个data_pack的日志序列
    log_sequences = []
    
    for data_pack in data_paths:
        # 获取相关文件路径
        fs = derive_filename(data_pack)
        
        # 检查必要文件是否存在
        if "abnormal_logs" not in fs or not os.path.exists(fs["abnormal_logs"]):
            raise FileNotFoundError(f"Abnormal log file not found for {data_pack.name}")
        if "injection" not in fs or not os.path.exists(fs["injection"]):
            raise FileNotFoundError(f"Injection file not found for {data_pack.name}")

        # 读取并预处理数据
        df = pd.read_parquet(fs["abnormal_logs"])
        df = df[df["service_name"] != "ts-ui-dashboard"].sort_values(by="time")
        
        # 只取前20条日志
        df = df.head(20)
        
        # 时间转换为毫秒级时间戳
        df["time"] = pd.to_datetime(df["time"], unit="s").astype(np.int64) // 10**6
        
        # 提取日志模板并计算模板ID
        df["template"] = df["message"].apply(drain)
        # df["template_id"] = df["template"].apply(lambda x: hex(abs(hash(x)))[2:10])
        df["template_id"] = df["template"].apply(generate_template_id)
        
        # 将当前data_pack的日志转换为[time, service_name, template_id]格式的列表
        log_sequence = df[["time", "service_name", "template_id"]].values.tolist()
        log_sequences.append(log_sequence)

    # 保存缓存
    drain.save_cache()
            
    # 保存为npy文件
    save_path = Path("/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/anomalies/stratification_logs.npy")
    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), np.array(log_sequences, dtype=object))
    
    print(f"已保存日志序列，形状: {len(log_sequences)}")
    print(f"第一个序列示例: {log_sequences[0][:3]}")
    
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