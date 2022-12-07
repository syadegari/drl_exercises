import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer
from models import QNetwork_Dueling


def maxQ(q_net, states):
    return q_net(states).cpu().detach().max(1)[0].unsqueeze(1)


def argmaxQ(q_net, states):
    return q_net(states).cpu().detach().argmax(1).unsqueeze(1)


def Q(q_net, states, actions):
    return q_net(states).gather(1, actions)


class Agent:

    def __init__(self,
                 algo,
                 state_size,
                 action_size,
                 seed,
                 update_every: int=4,
                 lr: float=1e-3,
                 tau: float=1e-3,
                 gamma: float=0.99,
                 batch_size: int=64,
                 buffer_size: int=int(1e5)
                 ):
        self.agent_rng = np.random.default_rng(seed)
        
        self.memory = ReplayBuffer(buffer_size, action_size, batch_size, seed)
        self.qnetwork_local = algo.model(state_size, action_size, seed)
        self.qnetwork_target = algo.model(state_size, action_size, seed)
            
        self.learn_fn = algo.learn_fn
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size

        self.update_every = update_every
        self.t_step = 0

        self.action_size = action_size

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def step(self, state, action, reward, next_state, done):
        #
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        #
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn(*self.memory.sample())
            self.soft_update()

    def soft_update(self):
        '''
        soft updates target parameters
        target <- tau * p_local + (1 - tau) * p_target
        where tau << 1
        '''
        for p_local, p_target in zip(
            self.qnetwork_local.parameters(),
            self.qnetwork_target.parameters()
        ):
            p_target.data.copy_(self.tau * p_local.data + \
                                (1 - self.tau) * p_target.data)

    def act(self, state, epsilon):
        '''
        Returns action for a given state

        Params
        ======
            state: current state
            epsilon: for epsilon-greedy algorithm
        '''
        if self.agent_rng.binomial(1, 1 - epsilon):
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(
                    torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                )
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        #
        return self.agent_rng.choice(self.action_size)

    def learn(self, states, actions, rewards, next_states, dones):
        loss = self.learn_fn(
            self.qnetwork_local,
            self.qnetwork_target,
            states,
            actions,
            rewards,
            next_states,
            dones,
            self.gamma
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
