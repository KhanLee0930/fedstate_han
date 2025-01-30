from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["TRANSFORMERS_CACHE"] = "/scratch/e1310988"
os.environ['HF_DATASETS_CACHE'] = "/scratch/e1310988"

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]