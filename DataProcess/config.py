import pandas as pd
import yaml
from pathlib import Path

def update_config_nodes():
    """
    从gt.csv中读取所有instance并更新gaia_config2.yaml中的nodes字段
    """
    # 读取gt.csv文件
    gt_path = Path("/home/nn/workspace/DiagFusion/data/gaia/demo/demo2/gt.csv")
    gt_df = pd.read_csv(gt_path)
    
    # 获取所有唯一的instance
    instances = gt_df["instance"].unique()
    # 将instances转换为空格分隔的字符串
    nodes_str = " ".join(instances)
    
    # 读取yaml配置文件
    config_path = Path("/home/nn/workspace/DiagFusion/config/gaia_config2.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新nodes字段
    # 更新parse.nodes
    config['parse']['nodes'] = nodes_str
    # 更新fasttext.nodes
    config['fasttext']['nodes'] = nodes_str
    # 更新he_dgl.nodes
    config['he_dgl']['nodes'] = nodes_str
    
    # 保存更新后的配置
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("配置文件已更新！")
    print(f"更新的nodes为: {nodes_str}")
