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
from src.preprocess.process_data import preprocess_injection_inference
from src.diagfusion.data.preprocessing import RawDataProcess
from src.exp.config import deal_config
from src.diagfusion.data.dataset import UnircaDataset
from src.exp.controller import UnircaLab


def platform_get_config():
    config_path = "./src/config/gaia_config2.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class diagfusion(Algorithm):
    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> list[AlgorithmAnswer]:
        # 输入参数为 输入 case 的路径
        # 输出预测故障服务的排位
        # ==========================================================
        logger.info(f"Processing case: {args.input_folder}")
        config = platform_get_config()

        # 处理数据
        # 预处理日志数据
        cache_dir = "./cache"
        data_paths = args.input_folder
        # =========================================
        # 处理 groundtruth 数据
        preprocess_injection_inference(data_paths)

        # =========================================

        # 保存相应的数据到对应目录
        # 输入：
        #     sentence_embedding.pkl 训练过程中得到的
        #     demo.csv
        he_dgl_config = deal_config(config, "he_dgl")
        RawDataProcess(he_dgl_config).process(inference=True)

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
            UnircaDataset(
                os.path.join(save_dir, "test_Xs.pkl"),
                os.path.join(save_dir, "test_ys_service.pkl"),
                os.path.join(save_dir, "topology.pkl"),
            ),
            "instance",
        )

        return results
