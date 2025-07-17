import os
import json
import pandas as pd
from src.preprocess.dataset.k_sigma import Ksigma
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import logger


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

        fs_injection = derive_filename_injection(data_pack)
        # 读取injection文件
        with open(fs_injection["injection"], "r") as f:
            injection = json.load(f)
        # 没找到 groundtruth 字段，跳过
        if "ground_truth" not in injection:
            continue

        instance_id = (
            injection["ground_truth"]["service"][1]
            if len(injection["ground_truth"]["service"]) > 1
            else injection["ground_truth"]["service"][0]
        )
        # instance_id = injection["ground_truth"]["service"][0]

        try:
            # 读取 parquet 文件
            df = pd.read_parquet(parquet_file)

            # 确保数据框包含必要的列
            if "time" not in df.columns or "value" not in df.columns:
                logger.warning(f"跳过 {data_pack.name}: 缺少必要的列")
                continue

            metric_name = df["metric"].iloc[0]  # 从数据中获取metric字段的值
            service_name = df["service_name"].iloc[0]
            # 对数据进行排序
            df = df.sort_values("time")

            # 获取时间范围
            start_ts = df["time"].min()
            end_ts = df["time"].max()

            # 使用 K-sigma 进行检测
            is_anomalous, anomaly_ts, anomaly_score = detector.detection(
                data=df, column="value", start_ts=start_ts, end_ts=end_ts
            )

            # 如果检测到异常,添加到对应case_id的列表中
            if is_anomalous:
                # 如果这个case_id还没有对应的列表,创建一个
                if str(case_id) not in case_dict:
                    case_dict[str(case_id)] = []

                # 添加异常数据
                case_dict[str(case_id)].append(
                    [int(anomaly_ts), instance_id, metric_name, float(anomaly_score)]
                )
                logger.info(
                    f"已处理完成 case {case_id}, 检测到异常，score: {anomaly_score:.4f}"
                )

        except Exception as e:
            logger.error(f"处理 {data_pack.name} 时出错: {str(e)}")
            continue

    # 设置输出路径并确保目录存在
    output_file = Path("src/data/rcabench/demo/demo2/anomalies/demo_metric.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(case_dict, f, indent=4)

    logger.success(f"处理完成，结果已保存到: {output_file}")
    logger.info(f"共处理了 {len(case_dict)} 个case")

    return case_dict


def process_parquet_files_inference(data_path: Path, output_path: Path):
    # 初始化 K-sigma 检测器
    detector = Ksigma()

    # 存储所有结果,按case_id分组
    case_dict = {}

    # 读取 parquet 文件
    parquet_file = os.path.join(data_path, "abnormal_metrics.parquet")
    if not os.path.exists(parquet_file):
        logger.error(f"指标数据文件不存在: {parquet_file}")
        return case_dict

    df = pd.read_parquet(parquet_file)

    metric_name = df["metric"].iloc[0]  # 从数据中获取metric字段的值
    # 对数据进行排序
    df = df.sort_values("time")

    # 获取时间范围
    start_ts = df["time"].min()
    end_ts = df["time"].max()

    # 使用 K-sigma 进行检测
    is_anomalous, anomaly_ts, anomaly_score = detector.detection(
        data=df, column="value", start_ts=start_ts, end_ts=end_ts
    )

    # 如果检测到异常,添加到对应case_id的列表中
    if is_anomalous:
        # 如果这个case_id还没有对应的列表,创建一个
        if str(0) not in case_dict:
            case_dict[str(0)] = []

        # 添加异常数据
        case_dict[str(0)].append([int(anomaly_ts), metric_name, float(anomaly_score)])
        logger.info(f"已处理完成 case {0}, 检测到异常，score: {anomaly_score:.4f}")

    # 设置输出路径并确保目录存在
    output_file = os.path.join(output_path, "demo_metric.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(case_dict, f, indent=4)

    logger.success(f"处理完成，结果已保存到: {output_file}")
    logger.info(f"共处理了 {len(case_dict)} 个case")

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
        "injection": sub_dir / "injection.json",
    }


def derive_filename_injection(data_pack: Path) -> dict:
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
