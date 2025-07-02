import os
from src.exp.controller import UnircaLab
from src.exp.config import deal_config, get_config
import pandas as pd
from src.diagfusion.data.preprocessing import run_parse, run_fasttext, run_sentence_embedding

if __name__ == "__main__":
    print("[diagfusion]")
    # diagfusion
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
    UnircaLab(deal_config(config, "he_dgl")).do_lab(lab_id)



