import os
import json
import numpy as np
import pandas as pd
from torch import tensor
import dgl.data.utils as U
import random
import fasttext
import numpy as np
import src.diagfusion.utils.public_function as pf
import math
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time
from src.utils.logger import logger


def run_parse(config, labels):
    trace = None
    metric = None
    logs = None
    if config["log_path"]:
        logs = np.load(config["log_path"], allow_pickle=True)
    if config["metric_path"]:
        with open(config["metric_path"], "r", encoding="utf8") as fp:
            metric = json.load(fp)
    if config["trace_path"]:
        with open(config["trace_path"], "r", encoding="utf8") as fp:
            trace = json.load(fp)
    if config["inference"] == False:
        metric_trace_log_parse(
            trace, metric, logs, labels, config["save_path"], config["nodes"]
        )
    else:
        metric_trace_log_parse_inference(
            trace, metric, logs, labels, config["save_path"], config["nodes"]
        )


# def run_parse_inference(data_path):
#     trace = None
#     metric = None
#     logs = None
#     logs = np.load(
#         os.path.join(data_path, "stratification_logs.npy"), allow_pickle=True
#     )
#     with open(os.path.join(data_path, "demo_metric.json"), "r", encoding="utf8") as fp:
#         metric = json.load(fp)
#     with open(os.path.join(data_path, "demo_trace.json"), "r", encoding="utf8") as fp:
#         trace = json.load(fp)
#     metric_trace_log_parse_inference(trace, metric, logs, data_path)


def run_fasttext(config, labels):
    # event embedding流程；基于数据增强
    start_ts = time.time()
    lab2 = FastTextLab(config, labels)
    lab2.do_lab()
    end_ts = time.time()
    logger.info("fasttext time used:", end_ts - start_ts, "s")


def run_fasttext_inference(config, labels):
    FastTextLab_inference(config, labels)


def run_sentence_embedding(config):
    sentence_embedding(
        config["source_path"],
        config["train_path"],
        config["test_path"],
        config["save_path"],
        config["K_S"],
    )


def run_sentence_embedding_inference():
    sentence_embedding_inference(
        "/home/nn/workspace/DiagFusion/src/data/middle/event_embedding.pkl",
        "/home/nn/workspace/DiagFusion/src/data/inference/demo/demo2/fasttext/temp/test.txt",
        "/home/nn/workspace/DiagFusion/src/data/inference/demo/demo2/sentence_embedding.pkl",
        1,
    )


class RawDataProcess:
    """用来处理原始数据的类
    参数
    ----------
    config: dict
        配置参数
        Xs: 多个图的特征向量矩阵
        data_dir: 数据和结果存放路径
        dataset: 数据集名称 可选['21aiops', 'gaia']
    """

    def __init__(self, config):
        self.config = config

    def process(self, inference=False):
        """用来获取并保存中间数据
        参数
        ----------
        inference: bool, 是否为推理模式

        输入：
            sentence_embedding.pkl
            demo.csv 或 inference.csv(推理模式)
        输出：
            训练集：
                train_Xs.pkl
                train_ys_service.pkl
            测试集：
                test_Xs.pkl
                test_ys_service.pkl
            拓扑：
                topology.pkl
        """
        if inference:
            # 推理模式下从固定路径读取
            run_table = pd.read_csv("src/data/inference/gt.csv", index_col=0)
        else:
            # 训练模式下从配置路径读取
            run_table = pd.read_csv(
                os.path.join(self.config["data_dir"], self.config["run_table"]),
                index_col=0,
            )
        Xs = U.load_info(os.path.join(self.config["data_dir"], self.config["Xs"]))
        Xs = np.array(Xs)

        # 只处理service标签
        service_labels = self.get_label("service", run_table)

        save_dir = self.config["save_dir"]
        train_index = np.where(run_table["data_type"].values == "train")
        test_index = np.where(run_table["data_type"].values == "test")
        train_size = len(train_index[0])
        # 保存特征向量，特征向量是先训练集后测试集
        U.save_info(os.path.join(save_dir, "train_Xs.pkl"), Xs[:train_size])
        U.save_info(os.path.join(save_dir, "test_Xs.pkl"), Xs[train_size:])
        # 保存标签
        U.save_info(
            os.path.join(save_dir, f"train_ys_service.pkl"),
            service_labels[train_index],
        )
        U.save_info(
            os.path.join(save_dir, f"test_ys_service.pkl"), service_labels[test_index]
        )
        # 保存拓扑
        topology = self.get_topology()
        U.save_info(os.path.join(save_dir, "topology.pkl"), topology)

    def get_label(self, label_type, run_table):
        """process() 中调用，用来获取label
        参数
        ----------
        label_type: str
            label的类型，可选：['service', 'anomaly_type']
        run_table: pd.DataFrame

        返回值
        ----------
        labels: torch.tensor()
            label列表
        """
        meta_labels = sorted(list(set(list(run_table[label_type]))))
        labels_idx = {
            label: idx for label, idx in zip(meta_labels, range(len(meta_labels)))
        }
        labels = np.array(
            run_table[label_type].apply(lambda label_str: labels_idx[label_str])
        )
        return labels

    def get_topology(self):
        """process() 中调用，用来获取topology"""
        dataset = self.config["dataset"]
        # 同质图
        if dataset == "rcabench":
            # 从json文件中读取拓扑结构
            topology_path = "/home/nn/workspace/DiagFusion/src/data/rcabench/demo/demo2/anomalies/instance_topology.json"
            with open(topology_path, "r") as f:
                topology_data = json.load(f)

            topology = (topology_data["source_nodes"], topology_data["target_nodes"])
        else:
            raise Exception()

        return topology


class InferenceDataProcess:
    """用来处理原始数据的类，仅用于推理
    参数
    ----------
    config: dict
        配置参数
        Xs: 特征向量矩阵
        data_dir: 数据和结果存放路径
        dataset: 数据集名称 可选['21aiops', 'gaia']
    """

    def __init__(self, config):
        self.config = config

    def process(self):
        """用来获取并保存中间数据，仅用于推理场景

        输入：
            sentence_embedding.pkl
        输出：
            测试集：
                test_Xs.pkl
            拓扑：
                topology.pkl
        """
        # 加载特征向量
        Xs = U.load_info(
            "/home/nn/workspace/DiagFusion/src/data/inference/demo/demo2/sentence_embedding.pkl"
        )
        Xs = np.array(Xs)

        save_dir = self.config["save_dir"]

        # 直接保存所有特征向量作为测试数据
        U.save_info(os.path.join(save_dir, "test_Xs.pkl"), Xs)

        # 保存拓扑
        topology = self.get_topology()
        U.save_info(os.path.join(save_dir, "topology.pkl"), topology)

    def get_topology(self):
        """process() 中调用，用来获取topology"""
        dataset = self.config["dataset"]
        # 同质图
        if dataset == "rcabench":
            # 从json文件中读取拓扑结构
            topology_path = "/home/nn/workspace/DiagFusion/src/data/rcabench/demo/demo2/anomalies/instance_topology.json"
            with open(topology_path, "r") as f:
                topology_data = json.load(f)

            topology = (topology_data["source_nodes"], topology_data["target_nodes"])
        else:
            raise Exception()

        return topology


class FastTextLab_inference:
    def __init__(self, config, cases, split=True):
        self.config = config
        self.cases = cases
        self.nodes = config["nodes"].split()
        self.node_labels = dict(zip(self.nodes, range(len(self.nodes))))
        self.train_data, self.test_data = self.prepare_data()

    def prepare_data(self):
        metric_trace_text_path = self.config["text_path"]
        temp_data = pf.load(metric_trace_text_path)
        train = self.cases[self.cases["data_type"] == "train"].index
        test = self.cases[self.cases["data_type"] == "test"].index
        total = self.cases.index
        self.save_to_txt(temp_data, train, self.config["train_path"])
        logger.info(f"===================\n")
        self.save_to_txt(temp_data, test, self.config["test_path"])
        with open(self.config["train_path"], "r") as f:
            data = f.read().splitlines()

        with open(self.config["train_path"], "r") as f:
            train_data = f.read().splitlines()
        with open(self.config["test_path"], "r") as f:
            test_data = f.read().splitlines()
        return train_data, test_data

    def save_to_txt(self, data: dict, keys, save_path):
        fillna = False
        with open(save_path, "w") as f:
            for case_id in keys:
                case_id = case_id if case_id in data.keys() else str(case_id)
                for node_info in data[case_id]:
                    text = data[case_id][node_info]
                    if isinstance(text, str):
                        text = text.replace("(", "").replace(")", "")
                        if fillna and len(text) == 0:
                            text = "None"
                        f.write(f"{text}\t__label__{self.node_labels[node_info[0]]}\n")
                    elif isinstance(text, list):
                        text = " ".join(text)
                        if fillna and len(text) == 0:
                            text = "None"
                        f.write(f"{text}\t__label__{self.node_labels[node_info[0]]}\n")
                    else:
                        raise Exception("type error")
        return


class FastTextLab:
    def __init__(self, config, cases, split=True):
        self.config = config
        self.cases = cases
        if self.config["supervised"]:
            self.method = fasttext.train_supervised
        else:
            self.method = fasttext.train_unsupervised
        self.nodes = config["nodes"].split()
        self.node_labels = dict(zip(self.nodes, range(len(self.nodes))))
        self.train_data, self.test_data = self.prepare_data()

    def prepare_data(self):
        metric_trace_text_path = self.config["text_path"]
        temp_data = pf.load(metric_trace_text_path)
        train = self.cases[self.cases["data_type"] == "train"].index
        test = self.cases[self.cases["data_type"] == "test"].index
        total = self.cases.index
        self.save_to_txt(temp_data, train, self.config["train_path"])
        logger.info(f"===================\n")
        self.save_to_txt(temp_data, test, self.config["test_path"])
        with open(self.config["train_path"], "r") as f:
            data = f.read().splitlines()

        with open(self.config["train_path"], "r") as f:
            train_data = f.read().splitlines()
        with open(self.config["test_path"], "r") as f:
            test_data = f.read().splitlines()
        return train_data, test_data

    def w2v_DA(self):
        da_train_data = self.train_data.copy()
        model = self.method(
            self.config["train_path"],
            dim=self.config["vector_dim"],
            minCount=self.config["minCount"],
            minn=0,
            maxn=0,
            epoch=self.config["epoch"],
        )
        random.seed(0)
        for node in self.nodes:
            sample_count = len(
                [
                    text
                    for text in self.train_data
                    if text.split("__label__")[-1] == str(self.node_labels[node])
                ]
            )
            if sample_count == 0:
                continue
            service_texts = [
                text
                for text in self.train_data
                if text.split("\t")[-1] == f"__label__{self.node_labels[node]}"
            ]
            loop = 0
            while sample_count < self.config["sample_count"]:
                loop += 1
                if loop >= 10 * self.config["sample_count"]:
                    break
                chosen_text, label = service_texts[
                    random.randint(0, len(service_texts) - 1)
                ].split("\t")
                chosen_text_splits = chosen_text.split()
                if len(chosen_text_splits) < self.config["minCount"]:
                    continue
                edit_event_ids = random.sample(
                    range(len(chosen_text_splits)), self.config["edit_count"]
                )
                for event_id in edit_event_ids:
                    nearest_event = model.get_nearest_neighbors(
                        chosen_text_splits[event_id]
                    )[0][-1]
                    chosen_text_splits[event_id] = nearest_event
                da_train_data.append(
                    " ".join(chosen_text_splits)
                    + f"\t__label__{self.node_labels[node]}"
                )
                sample_count += 1

        with open(self.config["train_da_path"], "w") as f:
            for text in da_train_data:
                f.write(text + "\n")

    def event_embedding_lab(self, data_path):
        model = self.method(
            data_path,
            dim=self.config["vector_dim"],
            minCount=self.config["minCount"],
            minn=0,
            maxn=0,
            epoch=self.config["epoch"],
        )
        event_dict = dict()
        for event in model.words:
            event_dict[event] = model[event]
        # 保存供推理使用
        pf.save(
            "/home/nn/workspace/DiagFusion/src/data/middle/event_embedding.pkl",
            event_dict,
        )
        return event_dict

    def save_to_txt(self, data: dict, keys, save_path):
        fillna = False
        with open(save_path, "w") as f:
            for case_id in keys:
                case_id = case_id if case_id in data.keys() else str(case_id)
                for node_info in data[case_id]:
                    text = data[case_id][node_info]
                    if isinstance(text, str):
                        text = text.replace("(", "").replace(")", "")
                        if fillna and len(text) == 0:
                            text = "None"
                        f.write(f"{text}\t__label__{self.node_labels[node_info[0]]}\n")
                    elif isinstance(text, list):
                        text = " ".join(text)
                        if fillna and len(text) == 0:
                            text = "None"
                        f.write(f"{text}\t__label__{self.node_labels[node_info[0]]}\n")
                    else:
                        raise Exception("type error")
        return

    def do_lab(self):
        self.w2v_DA()
        pf.save(
            self.config["save_path"],
            self.event_embedding_lab(self.config["train_da_path"]),
        )


def metric_trace_log_parse(trace, metric, logs, labels, save_path, nodes):
    if not metric is None:  # 去除np.inf数值的指标
        for k, v in metric.items():
            metric[k] = [x for x in v if not math.isinf(x[3])]

    if not logs is None:
        logs = list(logs)
        log = {x: [] for x in labels.index}
        if labels.index[-1] + 1 == len(log):
            for k, v in log.items():
                log[k] = logs[int(k)]
        else:
            count = 0
            for k, v in log.items():
                log[k] = logs[count]
                count += 1

    service_name = nodes.split()
    anomaly_service = list(labels["instance"])
    anomaly_type = list(labels["anomaly_type"])

    demo_metric = {x: {} for x in labels.index}
    k = 0
    for case_id, v in tqdm(demo_metric.items()):
        anomaly_service_name = anomaly_service[k]
        anomaly_service_type = anomaly_type[k]
        k += 1
        inner_dict_key = [
            (x, anomaly_service_type) if x == anomaly_service_name else (x, "[normal]")
            for x in service_name
        ]
        # 指标
        if not metric is None:
            demo_metric[case_id] = {
                x: [
                    [y[0], "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")]
                    for y in metric[str(case_id)]
                    if y[1] == x[0]
                ]
                for x in inner_dict_key
            }
        else:
            demo_metric[case_id] = {x: [] for x in inner_dict_key}
        # 调用链
        if not trace is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend(
                    [
                        [y[0], "{}_{}".format(y[1], y[2])]
                        for y in trace[str(case_id)]
                        if y[1] == inner_key[0] or y[2] == inner_key[0]
                    ]
                )
        # 日志
        if not logs is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend(
                    [[y[0], y[2]] for y in log[case_id] if y[1] == inner_key[0]]
                )
        for inner_key in inner_dict_key:
            temp = demo_metric[case_id][inner_key]
            sort_list = sorted(temp, key=lambda x: x[0])
            temp_list = [x[1] for x in sort_list]
            demo_metric[case_id][inner_key] = " ".join(temp_list)

    pf.save(save_path, demo_metric)


def metric_trace_log_parse_inference(trace, metric, logs, labels, save_path, nodes):
    if not logs is None:
        logs = list(logs)
        log = {x: [] for x in labels.index}
        if labels.index[-1] + 1 == len(log):
            for k, v in log.items():
                log[k] = logs[int(k)]
        else:
            count = 0
            for k, v in log.items():
                log[k] = logs[count]
                count += 1

    service_name = nodes.split()
    anomaly_service = list(labels["instance"])
    anomaly_type = list(labels["anomaly_type"])

    demo_metric = {x: {} for x in labels.index}
    k = 0
    for case_id, v in tqdm(demo_metric.items()):
        anomaly_service_name = anomaly_service[k]
        anomaly_service_type = anomaly_type[k]
        k += 1
        inner_dict_key = [
            (x, anomaly_service_type) if x == anomaly_service_name else (x, "[normal]")
            for x in service_name
        ]
        # 指标
        if not metric is None:
            demo_metric[case_id] = {
                x: [
                    [y[0], "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")]
                    for y in metric[str(case_id)]
                ]
                for x in inner_dict_key
            }
        else:
            demo_metric[case_id] = {x: [] for x in inner_dict_key}
        # 调用链
        if not trace is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend(
                    [[y[0], "{}_{}".format(y[1], y[2])] for y in trace[str(case_id)]]
                )
        # 日志
        if not logs is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend(
                    [[y[0], y[2]] for y in log[case_id]]
                )
        for inner_key in inner_dict_key:
            temp = demo_metric[case_id][inner_key]
            sort_list = sorted(temp, key=lambda x: x[0])
            temp_list = [x[1] for x in sort_list]
            demo_metric[case_id][inner_key] = " ".join(temp_list)

    pf.save(save_path, demo_metric)


# def metric_trace_log_parse_inference(trace, metric, logs, save_path):
#     if not logs is None:
#         logs = list(logs)
#         log = {"0": logs[0]}

#     demo_metric = {"0": {}}
#     for case_id, v in tqdm(demo_metric.items()):
#         inner_dict_key = [("service", "[normal]")]
#         # 指标
#         if not metric is None:
#             demo_metric[case_id] = {
#                 x: [
#                     [y[0], "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")]
#                     for y in metric[str(case_id)]
#                     if y[1] == x[0]
#                 ]
#                 for x in inner_dict_key
#             }
#         else:
#             demo_metric[case_id] = {x: [] for x in inner_dict_key}
#         # 调用链
#         if not trace is None:
#             for inner_key in inner_dict_key:
#                 demo_metric[case_id][inner_key].extend(
#                     [
#                         [y[0], "{}_{}".format(y[1], y[2])]
#                         for y in trace[str(case_id)]
#                         if y[1] == inner_key[0] or y[2] == inner_key[0]
#                     ]
#                 )
#         # 日志
#         if not logs is None:
#             for inner_key in inner_dict_key:
#                 demo_metric[case_id][inner_key].extend(
#                     [[y[0], y[2]] for y in log[case_id] if y[1] == inner_key[0]]
#                 )
#         for inner_key in inner_dict_key:
#             temp = demo_metric[case_id][inner_key]
#             sort_list = sorted(temp, key=lambda x: x[0])
#             temp_list = [x[1] for x in sort_list]
#             demo_metric[case_id][inner_key] = " ".join(temp_list)

#     logger.info(
#         f"save demo_metric to {os.path.join(save_path, 'stratification_texts.pkl')}"
#     )
#     pf.save(os.path.join(save_path, "stratification_texts.pkl"), demo_metric)


def read_text(path):
    text = []
    f = open(path, "r")
    line = f.readline()
    text.append(line.split("\t")[0])
    while line:
        line = f.readline()
        text.append(line.split("\t")[0])
    f.close()
    # 去最后的空串
    return text[:-1]


def sentence_embedding(file_dict, train_path, test_path, save_path, service_num):
    data_dict = pf.load(file_dict)

    train_text = read_text(train_path)
    test_text = read_text(test_path)
    vectorizer = CountVectorizer(
        lowercase=False, token_pattern=r"(?u)\b\S\S+"
    )  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    vec_train = vectorizer.fit_transform(train_text)
    tfidf_train = transformer.fit_transform(vec_train)

    # 保存训练好的vectorizer和transformer模型
    import joblib

    model_dir = "/home/nn/workspace/DiagFusion/src/data/middle"
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    transformer_path = os.path.join(model_dir, "transformer.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(transformer, transformer_path)
    logger.info(f"Saved vectorizer model to {vectorizer_path}")
    logger.info(f"Saved transformer model to {transformer_path}")

    # 预测
    vec_test = vectorizer.transform(test_text)
    tfidf_test = transformer.transform(vec_test)

    weight_train = (
        tfidf_train.toarray()
    )  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    weight_test = tfidf_test.toarray()

    word = vectorizer.get_feature_names_out()  # 获取词袋模型中的所有词语
    word_dict = {word[i]: i for i in range(len(word))}
    logger.info("len vectorizer words:", len(word_dict))
    logger.info("len fasttext words:", len(data_dict))
    logger.info(
        "dict(fasttext words) - dict(vectorizer words) = ",
        set(data_dict.keys() - set(word_dict.keys())),
    )
    logger.info(
        "dict(vectorizer words) - dict(fasttext words) = ",
        set(word_dict.keys() - set(data_dict.keys())),
    )

    train_embedding = tfidf_word_embedding(
        weight_train, data_dict, train_text, word_dict, service_num
    )
    test_embedding = tfidf_word_embedding(
        weight_test, data_dict, test_text, word_dict, service_num
    )

    train_embedding.extend(test_embedding)

    logger.info(
        "sentence_embedding shape:",
        f"{len(train_embedding)} * {len(train_embedding[0])} * {len(train_embedding[0][0])}",
    )
    pf.save(save_path, train_embedding)


def sentence_embedding_inference(file_dict, test_path, save_path, service_num):
    data_dict = pf.load(file_dict)

    test_text = read_text(test_path)

    # 加载预先训练好的vectorizer和transformer模型
    import joblib

    vectorizer_path = "/home/nn/workspace/DiagFusion/src/data/middle/vectorizer.joblib"
    transformer_path = (
        "/home/nn/workspace/DiagFusion/src/data/middle/transformer.joblib"
    )

    try:
        vectorizer = joblib.load(vectorizer_path)
        transformer = joblib.load(transformer_path)
        logger.info(f"Loaded vectorizer model from {vectorizer_path}")
        logger.info(f"Loaded transformer model from {transformer_path}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError("请先运行训练过程以生成vectorizer和transformer模型")

    # 预测
    vec_test = vectorizer.transform(test_text)
    tfidf_test = transformer.transform(vec_test)

    weight_test = tfidf_test.toarray()

    word = vectorizer.get_feature_names_out()  # 获取词袋模型中的所有词语
    word_dict = {word[i]: i for i in range(len(word))}
    logger.info("len vectorizer words:", len(word_dict))
    logger.info("len fasttext words:", len(data_dict))

    test_embedding = tfidf_word_embedding(
        weight_test, data_dict, test_text, word_dict, service_num
    )

    pf.save(save_path, test_embedding)


def tfidf_word_embedding(weight, data_dict, texts, word_dict, service_num):
    length = len(data_dict[list(data_dict.keys())[0]])
    count = 0
    case_embedding = []
    sentence_embedding = []
    for text in texts:
        temp = np.array([0] * length, "float32")
        count_log = count_metric = count_trace = 0
        if text != "":
            words = list(set(text.split(" ")))
            #             words = text.split()
            for word in words:
                if word in word_dict:
                    temp = temp + weight[count][word_dict[word]] * np.array(
                        data_dict[word]
                    )
        case_embedding.append(temp)
        if (count + 1) % service_num == 0:  # @
            sentence_embedding.append(case_embedding)
            case_embedding = []
        count += 1
    return sentence_embedding
