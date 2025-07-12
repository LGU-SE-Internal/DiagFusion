import os
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
from src.diagfusion.models.layers import *
import warnings
import json
from src.diagfusion.data.dataset import UnircaDataset
from src.diagfusion.data.preprocessing import RawDataProcess
from src.utils.logger import logger

warnings.filterwarnings("ignore")


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
            mapping_path = "/home/nn/workspace/DiagFusion/src/data/rcabench/demo/demo2/anomalies/service_instance_mapping.json"
            with open(mapping_path, "r") as f:
                mapping_data = json.load(f)
                # 将service_instance_map的键转换为整数
                self.topoinfo = {
                    int(k): v for k, v in mapping_data["service_instance_map"].items()
                }
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
            model = RGCNClassifier(in_dim, hid_dim, out_dim, etype).to(device)
        else:
            model = TAGClassifier(in_dim, hid_dim, out_dim).to(device)
        logger.info(model)

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

    def inference(self, model, dataset, task, out_file=None, save_file=None):
        from rcabench_platform.v2.algorithms.spec import AlgorithmAnswer
        from src.utils.logger import logger

        model.eval()
        dataloader = DataLoader(
            dataset, batch_size=len(dataset) + 10, collate_fn=self.collate
        )
        device = "cpu"
        results = []

        # 创建实例ID到名称的反向映射
        id_to_name = {v: k for k, v in self.ins_dict.items()}

        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            output = model(batched_graph, batched_graph.ndata["attr"].float())

            if task == "instance":
                k = 5 if output.shape[-1] >= 5 else output.shape[-1]
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)
                predicted_services = indices.detach().numpy()[0]  # 获取预测的服务IDs

                # 获取实例并生成结果
                instances = []
                for service_id in predicted_services:
                    if service_id in self.topoinfo:
                        for instance_id in self.topoinfo[service_id]:
                            instances.append(instance_id)
                            if len(instances) >= 5:  # 最多返回5个实例
                                break
                    if len(instances) >= 5:
                        break

                # 构建AlgorithmAnswer结果
                for rank, instance_id in enumerate(instances, 1):
                    # 将实例ID转换为实例名称
                    instance_name = id_to_name.get(
                        instance_id, f"unknown_{instance_id}"
                    )
                    results.append(
                        AlgorithmAnswer(
                            level="service",
                            name=instance_name,  # 使用实例名称而不是ID
                            rank=rank,
                        )
                    )
            else:
                raise Exception("Unknow task")

        # 打印返回结果
        logger.info("返回的AlgorithmAnswer列表:")
        for idx, answer in enumerate(results):
            logger.info(
                f"  {idx+1}. level: {answer.level}, name: {answer.name}, rank: {answer.rank}"
            )

        return results

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
                ins_pred.extend([item[0] for item in temp[:max_num]])
            ins_preds.append(ins_pred[:5])
            for k in range(5):
                if ins_pred[k] == self.ins_dict[row["instance"]]:
                    topks[k:] += 1
                    break
            i += 1
        logger.info("Top1-5: {}".format(topks / len(test_cases)))
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
            "N_S",
        )

        logger.info("instance")
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
