import os
import torch
import yaml
import pandas as pd
from rcabench_platform.v2.algorithms.spec import (
    Algorithm,
    AlgorithmArgs,
    AlgorithmAnswer,
)
from src.utils.logger import logger
from src.diagfusion.data.preprocessing import InferenceDataProcess
from src.exp.config import deal_config
from src.diagfusion.data.dataset import InferenceDataset
from src.exp.controller import UnircaLab
from src.preprocess.process_data import process_data_inference


class diagfusion(Algorithm):
    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> list[AlgorithmAnswer]:
        # 输入参数为 输入 case 的路径
        # 输出预测故障服务的排位
        # ==========================================================
        logger.info(f"Processing case: {args.input_folder}")

        # 处理数据
        # 预处理日志数据
        cache_dir = "./cache"
        data_paths = args.input_folder

        process_data_inference(data_paths, cache_dir="./cache")

        config = yaml.safe_load(
            open("/home/nn/workspace/DiagFusion/src/config/gaia_config2.yaml")
        )

        # 保存相应的数据到对应目录
        # 输入：
        #     sentence_embedding.pkl
        he_dgl_config = deal_config(config, "he_dgl")
        InferenceDataProcess(he_dgl_config).process()

        # 从环境变量获取模型文件路径
        model_path = os.getenv("CHECKPOINT_PATH")
        if model_path is None:
            raise ValueError("未设置CHECKPOINT_PATH环境变量")

        # 加载模型
        model_ts = torch.load(model_path)

        logger.info("start inference")

        lab_id = 9
        save_dir = os.path.join(he_dgl_config["save_dir"], str(lab_id))

        # 创建UnircaLab实例
        lab = UnircaLab(he_dgl_config)

        # inference函数现在直接返回list[AlgorithmAnswer]
        results = lab.inference(
            model_ts,
            InferenceDataset(
                os.path.join(save_dir, "test_Xs.pkl"),
                os.path.join(save_dir, "topology.pkl"),
            ),
            "instance",
        )

        return results
