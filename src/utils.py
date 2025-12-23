import os 
import random
import numpy as np
from src.exception import CustomException
import sys

import torch

def set_seed(seed :int):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.maual_seed_all(seed)
    except Exception:
        pass

def info(msg: str):
    print(f"[INFO]{msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")


def error(msg: str):
    print(f"[ERROR] {msg}")
