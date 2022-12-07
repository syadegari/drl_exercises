import os
from itertools import product
from dqn_mine import dqn
from agent import Agent
from dataclasses import dataclass
from typing import Callable, Any
import pickle


@dataclass
class State:
    agent: Agent
    env: Env
    i_episode: int


N_EPISODES = 200


def init_from_zero(env, seed):
    env.reset(seed=seed)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    agent = Agent(nS, nA, seed)

    return env, agent


def load(loadname: str):
    with open(loadname, "rb") as fh:
        state = pickle.load(fh)
    return state


def save(savename: str, state: State):
    with open(savename, "wb") as fh:
        pickle.dump(state, fh)


def goal_fn(env, scores_window, state, i_episode):
    if i_episode > 2000:
        return True
    return False


from dqn_mine import dqn
from dqn_algorithms import DQN, DDQN, DuelingDDQN
import gym


env_names = ["CartPole-v1", "MountainCar-v0", "LunarLander-v2"]
envs = [gym.make(env_name) for env_name in env_names]

algos = [DQN, DDQN, DuelingDDQN]


# We run everything for a fixed number of episodes and don't care about task completion.
# This way we don't have to worry about the completion of different environements.
def main():
    seed = 101
    repeats = range(1, 3)
    for env, algo, repeat in product(envs, algos, repeats):
        dqn(env, seed, algo, , repeat, N_EPISODES)


if __name__ == "__main__":
    main()
