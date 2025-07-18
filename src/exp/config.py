import os
import pickle
import argparse
import yaml


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    with open(os.path.join("./src/config", args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def deal_config(config, key):
    new_config = {}
    for k in config[key].keys():
        if "path" in k or "dir" in k:
            if config[key][k] or config[key][k] == "":
                path = os.path.join(
                    config["base_path"],
                    config["demo_path"],
                    config["label"],
                    config[key][k],
                )
                if "dir" in k:
                    if not os.path.exists(path):
                        os.makedirs(path)
                new_config[k] = path
            else:
                new_config[k] = config[key][k]
        else:
            new_config[k] = config[key][k]

    return new_config
