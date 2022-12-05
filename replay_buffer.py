from collections import deque
from dataclasses import dataclass
import numpy as np
from numpy import vstack
import torch
from torch import from_numpy
from numba import njit


@njit
def collect_idx_0(experiences):
    return [e[0] for e in experiences if e is not None]

@njit
def collect_idx_1(experiences):
    return [e[1] for e in experiences if e is not None]

@njit
def collect_idx_2(experiences):
    return [e[2] for e in experiences if e is not None]

@njit
def collect_idx_3(experiences):
    return [e[3] for e in experiences if e is not None]

@njit
def collect_idx_4(experiences):
    return [e[4] for e in experiences if e is not None]


@dataclass
class ReplayBuffer:
    buffer_size: int
    action_size: int
    batch_size: int
    seed: int

    def __post_init__(self):
        self.seed = np.random.seed(self.seed)
        self.memory = deque(maxlen=self.buffer_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(
            (
                state,               # 0
                np.int32(action),    # 1
                np.float32(reward),  # 2
                next_state,          # 3
                done                 # 4
            )
        )

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = np.random.choice(self.memory, self.batch_size, replace=False)
        #
        states = from_numpy(vstack(collect_idx_0(experiences))).float().to(self.device)
        actions = from_numpy(vstack(collect_idx_1(experiences))).long().to(self.device)
        rewards = from_numpy(vstack(collect_idx_2(experiences))).float().to(self.device)
        next_states = from_numpy(vstack(collect_idx_3(experiences))).float().to(self.device)
        dones = from_numpy(vstack(collect_idx_4(experiences)).astype(np.uint8)).float().to(self.device)
        #
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


        
