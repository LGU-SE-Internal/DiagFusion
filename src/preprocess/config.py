import pandas as pd
import yaml
from pathlib import Path
from src.utils.logger import logger


def update_config_nodes():
    """
    从gt.csv中读取所有instance和service，更新配置文件中的相关字段：
    1. 更新nodes字段为所有instance
    2. 更新N_S为service的数量
    3. 更新K_S为instance的数量
    """
    # 读取gt.csv文件
    gt_path = Path("src/data/rcabench/demo/demo2/gt.csv")
    # 确保gt.csv所在目录存在
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_df = pd.read_csv(gt_path)

    # 获取所有唯一的instance和service
    instances = gt_df["instance"].unique()
    services = gt_df["service"].unique()

    # 计算数量
    instance_count = len(instances)
    service_count = len(services)

    # 将instances转换为空格分隔的字符串
    nodes_str = " ".join(instances)

    # 读取yaml配置文件
    config_path = Path("src/config/gaia_config2.yaml")
    # 确保配置文件所在目录存在
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 更新nodes字段
    config["parse"]["nodes"] = nodes_str
    config["fasttext"]["nodes"] = nodes_str
    config["he_dgl"]["nodes"] = nodes_str

    # 更新N_S为service数量
    config["he_dgl"]["N_S"] = service_count
    config["sentence_embedding"]["K_S"] = instance_count

    # 保存更新后的配置
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.success("配置文件已更新！")
    logger.info(f"更新的nodes为: {nodes_str}")
    logger.info(f"更新的N_S（服务数量）为: {service_count}")
    logger.info(f"更新的K_S（instance数量）为: {instance_count}")
