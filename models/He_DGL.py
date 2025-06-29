import os
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import dgl
import dgl.data.utils as U
import time
import pickle
from models.layers import *
import warnings
import json

warnings.filterwarnings("ignore")


class UnircaDataset:
    """
    参数
    ----------
    dataset_path: str
        数据存放位置。
        举例: 'train_Xs.pkl' （67 * 14 * 40）（图数 * 节点数 * 节点向量维数）
    labels_path: str
        标签存放位置。
        举例: 'train_ys_anomaly_type.pkl' （67）
    topology: str
        图的拓扑结构存放位置
        举例：'topology.pkl'
    aug: boolean (default: False)
        需要数据增强，该值设置为True
    aug_size: int (default: 0)
        数据增强时，每个label对应的样本数
    shuffle: boolean (default: False)
        load()完成以后，若shuffle为True，则打乱self.graphs 和 self.labels （同步）
    """

    def __init__(
        self, dataset_path, labels_path, topology, aug=False, aug_size=0, shuffle=False
    ):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.topology = topology
        self.aug = aug
        self.aug_size = aug_size
        self.graphs = []
        self.labels = []
        self.load()
        if shuffle:
            self.shuffle()

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def load(self):
        """__init__()  中使用，作用是装载 self.graphs 和 self.labels，若aug为True，则进行数据增强操作。"""
        Xs = tensor(U.load_info(self.dataset_path))
        ys = tensor(U.load_info(self.labels_path))
        topology = U.load_info(self.topology)
        assert Xs.shape[0] == ys.shape[0]
        if self.aug:
            Xs, ys = self.aug_data(Xs, ys)

        for X in Xs:
            g = dgl.graph(topology)  # 同质图
            # 若有0入度节点，给这些节点加自环
            in_degrees = g.in_degrees()
            zero_indegree_nodes = [
                i for i in range(len(in_degrees)) if in_degrees[i].item() == 0
            ]
            for node in zero_indegree_nodes:
                g.add_edges(node, node)

            g.ndata["attr"] = X
            self.graphs.append(g)
        self.labels = ys

    def shuffle(self):
        graphs_labels = [(g, l) for g, l in zip(self.graphs, self.labels)]
        random.shuffle(graphs_labels)
        self.graphs = [i[0] for i in graphs_labels]
        self.labels = [i[1] for i in graphs_labels]

    def aug_data(self, Xs, ys):
        """load() 中使用，作用是数据增强
        参数
        ----------
        Xs: tensor
            多个图对应的特征向量矩阵。
            举例：67个图对应的Xs规模为 67 * 14 * 40 （67个图，每个图14个节点）
        ys: tensor
            每个图对应的label，要求是从0开始的整数。
            举例：如果一共有10个label，那么ys中元素值为 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        self.aug_size: int
            数据增强时，每个label对应的样本数

        返回值
        ----------
        aug_Xs: tensor
            数据增强的结果
        aug_ys: tensor
            数据增强的结果
        """
        aug_Xs = []
        aug_ys = []
        num_label = len(set([y.item() for y in ys]))
        grouped_Xs = [[] for i in range(num_label)]
        for X, y in zip(Xs, ys):
            grouped_Xs[y.item()].append(X)
        for group_idx in range(len(grouped_Xs)):
            cur_Xs = grouped_Xs[group_idx]
            n = len(cur_Xs)
            m = Xs.shape[1]
            while len(cur_Xs) < self.aug_size:
                select = np.random.choice(n, m)
                aug_X = torch.zeros_like(Xs[0])
                for i, j in zip(select, range(m)):
                    aug_X[j] = cur_Xs[i][j].detach().clone()
                cur_Xs.append(aug_X)
            for X in cur_Xs:
                aug_Xs.append(X)
                aug_ys.append(group_idx)
        aug_Xs = torch.stack(aug_Xs, 0)
        aug_ys = tensor(aug_ys)
        return aug_Xs, aug_ys


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

    def process(self):
        """用来获取并保存中间数据
        输入：
            sentence_embedding.pkl
            demo.csv
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
        run_table = pd.read_csv(
            os.path.join(self.config["data_dir"], self.config["run_table"]), index_col=0
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
        # 保存边的类型(异质图)
        if self.config["heterogeneous"]:
            edge_types = self.get_edge_types()
            U.save_info(os.path.join(save_dir, "edge_types.pkl"), edge_types)

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
        if self.config["heterogeneous"]:
            # 异质图
            if dataset == "gaia":
                topology = (
                    [8, 6, 8, 4, 6, 4, 2, 9, 1, 3, 3, 7, 1, 7, 5, 0, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 2, 2, 3, 3, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
                    [6, 8, 4, 8, 4, 6, 9, 2, 3, 1, 7, 3, 7, 1, 0, 5, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 2, 2, 3, 3],
                )
            else:
                raise Exception()
        else:
            # 同质图
            if dataset == "rcabench":
                # 从json文件中读取拓扑结构
                topology_path = "/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/anomalies/instance_topology.json"
                with open(topology_path, 'r') as f:
                    topology_data = json.load(f)
                
                topology = (
                    topology_data["source_nodes"],
                    topology_data["target_nodes"]
                )
            else:
                raise Exception()
        return topology

    def get_edge_types(self):
        dataset = self.config["dataset"]
        if not self.config["heterogeneous"]:
            raise Exception()
        if dataset == "gaia":
            etype = tensor(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                        2,
                    ]
                ).astype(np.int64)
            )
        else:
            raise Exception()
        return etype


class UnircaLab:
    def __init__(self, config):
        self.config = config
        instances = config["nodes"].split()
        self.ins_dict = dict(zip(instances, range(len(instances))))
        self.demos = pd.read_csv(
            os.path.join(self.config["data_dir"], self.config["run_table"]), index_col=0
        )
        if config["dataset"] == "rcabench":
            # 从service_instance_mapping.json读取拓扑信息
            mapping_path = "/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/anomalies/service_instance_mapping.json"
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)
                # 将service_instance_map的键转换为整数
                self.topoinfo = {int(k): v for k, v in mapping_data["service_instance_map"].items()}
        else:
            raise Exception("Unknow dataset")

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

    def save_result(self, save_path, data):
        df = pd.DataFrame(data, columns=["top_k", "accuracy"])
        df.to_csv(save_path, index=False)

    def train(self, dataset, key):
        if self.config["seed"] is not None:
            torch.manual_seed(self.config["seed"])
        dataloader = DataLoader(
            dataset, batch_size=self.config["batch_size"], collate_fn=self.collate
        )
        device = "cpu"

        in_dim = dataset.graphs[0].ndata["attr"].shape[1]
        out_dim = self.config[key]
        # hid_dim = (in_dim + out_dim) * 2 // 3
        hid_dim = int(np.sqrt(in_dim * out_dim))
        if self.config["heterogeneous"]:
            etype = U.load_info(os.path.join(self.config["save_dir"], "edge_types.pkl"))
            model = RGCNClassifier(in_dim, hid_dim, out_dim, etype).to(
                device
            )
        else:
            model = TAGClassifier(in_dim, hid_dim, out_dim).to(device)
        print(model)

        opt = torch.optim.Adam(
            model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        losses = []
        model.train()
        for epoch in tqdm(range(self.config["epoch"])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for batched_graph, labels in dataloader:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                feats = batched_graph.ndata["attr"].float()
                logits = model(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if (
                len(losses) > self.config["win_size"]
                and abs(losses[-self.config["win_size"]] - losses[-1])
                < self.config["win_threshold"]
            ):
                break
        return model

    def testv2(self, model, dataset, task, out_file, save_file=None):
        model.eval()
        dataloader = DataLoader(
            dataset, batch_size=len(dataset) + 10, collate_fn=self.collate
        )
        device = "cpu"
        seed = self.config["seed"]
        accuracy = []
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            output = model(batched_graph, batched_graph.ndata["attr"].float())
            k = 5 if output.shape[-1] >= 5 else output.shape[-1]
            if task == "instance":
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)
                out_dir = os.path.join(self.config["save_dir"], "preds")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y_pred = indices.detach().numpy()
                y_true = labels.detach().numpy().reshape(-1, 1)
                ser_res = pd.DataFrame(
                    np.append(y_pred, y_true, axis=1),
                    columns=np.append(
                        [f"Top{i}" for i in range(1, len(y_pred[0]) + 1)], "GroundTruth"
                    ),
                )

                # 定位到实例级别
                accs, ins_res = self.test_instance_local(ser_res, max_num=2)
                ins_res.to_csv(f"{out_dir}/service_seed{seed}_{out_file}")
                columns = ["A@1", "A@2", "A@3", "A@4", "A@5"]
            else:
                raise Exception("Unknow task")

        if save_file:
            accuracy = pd.DataFrame(accs.reshape(-1, len(columns)), columns=columns)
            save_dir = os.path.join(
                self.config["save_dir"], "evaluations", save_file.split("_")[0]
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f"{save_dir}/seed{seed}_{save_file}", accuracy)

        return output, labels


    def test_instance_local(self, s_preds, max_num=2):
        """
        根据微服务的预测结果预测微服务的根因实例
        """
        with open(self.config["text_path"], "rb") as f:
            info = pickle.load(f)
        ktype = type(list(info.keys())[0])
        test_cases = self.demos[self.demos["data_type"] == "test"]
        topks = np.zeros(5)
        ins_preds = []
        i = 0
        for index, row in test_cases.iterrows():
            index = ktype(index)
            num_dict = {}
            for pair in info[index]:
                num_dict[self.ins_dict[pair[0]]] = len(info[index][pair].split())
            s_pred = s_preds.loc[i]
            ins_pred = []
            for col in list(s_preds.columns)[:-1]:
                temp = sorted(
                    [
                        (ins_id, num_dict[ins_id])
                        for ins_id in self.topoinfo[s_pred[col]]
                    ],
                    key=lambda x: x[-1],
                    reverse=True,
                )
                # print(self.topoinfo[s_pred[col]], temp)
                ins_pred.extend([item[0] for item in temp[:max_num]])
            ins_preds.append(ins_pred[:5])
            for k in range(5):
                if ins_pred[k] == self.ins_dict[row["instance"]]:
                    topks[k:] += 1
                    break
            i += 1
        print("Top1-5: ", topks / len(test_cases))
        y_true = np.array(
            [self.ins_dict[ins] for ins in test_cases["instance"].values]
        ).reshape(-1, 1)
        return topks / len(test_cases), pd.DataFrame(
            np.append(ins_preds, y_true, axis=1),
            columns=["Top1", "Top2", "Top3", "Top4", "Top5", "GroundTruth"],
            index=test_cases.index,
        )


    def do_lab(self, lab_id):
        save_dir = os.path.join(self.config["save_dir"], str(lab_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.config["save_dir"] = save_dir
        RawDataProcess(self.config).process()
        # 训练
        s = time.time()

        # 只训练service模型
        model_ts = self.train(
            UnircaDataset(
                os.path.join(save_dir, "train_Xs.pkl"),
                os.path.join(save_dir, "train_ys_service.pkl"),
                os.path.join(save_dir, "topology.pkl"),
                aug=self.config["aug"],
                aug_size=self.config["aug_size"],
                shuffle=True,
            ),
            "N_S"
        )
        
        print("instance")
        _, _ = self.testv2(
            model_ts,
            UnircaDataset(
                os.path.join(save_dir, "test_Xs.pkl"),
                os.path.join(save_dir, "test_ys_service.pkl"),
                os.path.join(save_dir, "topology.pkl"),
            ),
            "instance",
            "instance_pred_service.csv",
            "instance_acc_service.csv",
        )
        # 保存模型
        if self.config["save_model"]:
            torch.save(model_ts, os.path.join(save_dir, "service_model.pt"))
