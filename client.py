import os

from src.diagfusion.models import He_DGL
from src.diagfusion.utils.public_function import deal_config, get_config
import pandas as pd

import json
import math
from tqdm import tqdm
from src.diagfusion.data.preprocessing import metric_trace_log_parse

import numpy as np
import public_function as pf
import time
from src.diagfusion.data.preprocessing import FastTextLab
from src.diagfusion.data.preprocessing import sentence_embedding


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
    metric_trace_log_parse(
        trace, metric, logs, labels, config["save_path"], config["nodes"]
    )

def run_fasttext(config, labels):
    # event embedding流程；基于数据增强
    start_ts = time.time()
    lab2 = FastTextLab(config, labels)
    lab2.do_lab()
    end_ts = time.time()
    print("fasttext time used:", end_ts - start_ts, "s")

def run_sentence_embedding(config):
    sentence_embedding(
        config["source_path"],
        config["train_path"],
        config["test_path"],
        config["save_path"],
        config["K_S"],
    )





if __name__ == "__main__":
    config = get_config()
    label_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )
    labels = pd.read_csv(label_path, index_col=0)

    print('[parse]')
    run_parse(deal_config(config, 'parse'), labels)

    print("[fasttext]")
    run_fasttext(deal_config(config, "fasttext"), labels)

    print("[sentence_embedding]")
    run_sentence_embedding(
        deal_config(config, "sentence_embedding")
    )

    print("[dgl]")
    lab_id = 9  # 实验唯一编号
    He_DGL.UnircaLab(deal_config(config, "he_dgl")).do_lab(lab_id)



