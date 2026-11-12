"""公共工具：随机种子、目录、日志。"""
import os
import random
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(ROOT, "results")
RUN_DIR = os.path.join(RESULT_DIR, "runs")
FIG_DIR = os.path.join(ROOT, "figures")


def ensure_dirs():
    for d in (DATA_DIR, RESULT_DIR, RUN_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
