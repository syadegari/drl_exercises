from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import QNetwork, QNetwork_Dueling
from learn_fns import learn_dqn_with_target, learn_ddqn_with_target

@dataclass
class DQNAlgorithm:
    name: str
    learn_fn : Callable
    model: nn.Module
    comment: str = ''


DQN = DQNAlgorithm(
    name='DQN',
    learn_fn=learn_dqn_with_target,
    model=QNetwork,
    comment='Vanilla DQN Minh et al. (2015)'
)

DDQN = DQNAlgorithm(
    name='DDQN',
    learn_fn=learn_ddqn_with_target,
    model=QNetwork,
    comment='Double DQN DDQN Hesselt et al. (2015)'
)

Dueling_DDQN = DQNAlgorithm(
    name='Dueling_DDQN',
    learn_fn=learn_ddqn_with_target,
    model=QNetwork_Dueling,
    comment='Combining DDQN (Hasselt et al 2015) and Dueling DQN (Wang et al 2016)'        
)


