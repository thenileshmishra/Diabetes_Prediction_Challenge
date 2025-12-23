import os
import random
import logging
import numpy as np
from src.exception import CustomException
import sys

import torch

logger = logging.getLogger(__name__)

def set_seed(seed :int):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def info(msg: str):
    logger.info(msg)

def warn(msg: str):
    logger.warning(msg)


def error(msg: str):
    logger.error(msg)
