import os
import random
import torch
import numpy as np


def set_seed(seed_value=1234):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

