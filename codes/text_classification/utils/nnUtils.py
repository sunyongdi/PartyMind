import torch
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Union


logger = logging.getLogger(__name__)

__all__ = [
    'manual_seed',
    # 'seq_len_to_mask',
    # 'to_one_hot',
]


def manual_seed(seed: int = 1) -> None:
    """
        设置seed。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False