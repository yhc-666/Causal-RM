import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import importlib
import inspect
import json
import shutil
import torch
import yaml

from argparse import ArgumentParser
from safetensors.torch import load_file, save_file


def parse_arguments():
    parser = ArgumentParser(description="")
    parser.add_argument("--src_model_dir", type=str, default="/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/ckpts/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--src_model_class", type=str, default="llama")
    parser.add_argument("--rm_model_dir", type=str, default="/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/TrustworthRLHF/causal-rm/results/cache/ips/saferlhf")
    parser.add_argument("--rm_model_class", type=str, default="ips")
    parser.add_argument("--output_dir", type=str, default="/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/ckpts/IPS-RM")
    args = parser.parse_args()
    return args


def load_class_from_file(file_path, class_name):
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot create spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"'{class_name}' not found in '{file_path}'")

    return cls


def build_from_config(cls, cfg: dict):
    # 拿到 __init__ 的参数名（去掉 self）
    params = inspect.signature(cls.__init__).parameters
    names = [k for k in params.keys() if k != "self"]
    # 从 cfg 里按名字挑出有的键
    kwargs = {k: cfg[k] for k in names if k in cfg}
    # 用挑出来的参数实例化
    return cls(**kwargs)


def main():
    args = parse_arguments()

    # copy src model
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copytree(args.src_model_dir, args.output_dir, dirs_exist_ok=True)

    # copy rm template
    rm_template_dir = os.path.join(ROOT, "template", args.src_model_class)
    for filename in ["configuration_myrm.py", "modeling_myrm.py"]:
        shutil.copy2(os.path.join(rm_template_dir, filename), os.path.join(args.output_dir, filename))

    # load configs
    src_model_config = json.load(open(os.path.join(args.output_dir, "config.json"), "r"))
    rm_model_config = yaml.safe_load(open(os.path.join(args.rm_model_dir, "config.yaml"), "r"))
    rm_model_config["input_size"] = src_model_config["hidden_size"]
    rm_model_config["hidden_dim_str"] = rm_model_config["hidden_dim"]

    # load rm model
    rm_model_class = load_class_from_file(f"{ROOT}/causal-rm/benchmark_{args.rm_model_class}.py", "Model")
    rm_model = build_from_config(rm_model_class, rm_model_config)
    rm_model.load_state_dict(torch.load(os.path.join(args.rm_model_dir, "best_model.pth"), map_location="cpu"))
    state_dict = rm_model.state_dict()
    state_dict = {f"myscore.{k}": v for k, v in state_dict.items()}
    save_file(state_dict, os.path.join(args.output_dir, "myrm.safetensors"))

    # save config
    src_model_config["hidden_dim_str"] = rm_model_config["hidden_dim_str"]
    src_model_config["auto_map"] = {
        "AutoConfig": "configuration_myrm.MyRMConfig",
        "AutoModelForTokenClassification": "modeling_myrm.MyRMForTokenClassification"
    }
    src_model_config["_name_or_path"] = args.output_dir
    json.dump(src_model_config, open(os.path.join(args.output_dir, "config.json"), "w"), indent=2, sort_keys=True)

    # save index
    src_model_index = json.load(open(os.path.join(args.output_dir, "model.safetensors.index.json"), "r"))
    rm_model_size = sum([v.numel() for v in state_dict.values()])
    src_model_index["metadata"]["total_size"] += rm_model_size
    rm_params_map = {k: "myrm.safetensors" for k in state_dict.keys()}
    src_model_index["weight_map"].update(rm_params_map)
    json.dump(src_model_index, open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w"), indent=2)

    print("Next to do:")
    print("1. check the configuration_myrm.py to add `hidden_dim_str` or other parameters")
    print("2. check the modeling_myrm.py to add `myscore`")


if __name__ == '__main__':
    main()
