import os
from collections import deque
import pickle
from typing import Any
from dataclasses import dataclass, field

from agent import Agent

import gym
import torch
import numpy as np


@dataclass
class State:
    agent: Agent
    env: Any
    i_episode: int
    eps: float
    scores: list = field(default_factory=list)
    scores_window: deque = field(default_factory=deque)


def get_experiment_name(env_name, algo, repeat):
    return f"{env_name}_{algo.name}_{repeat}"


def init_env(env_name, seed):
    env = gym.make(env_name)
    env.reset(seed=seed)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    return env, nA, nS


def print_info(i_episode, print_every, scores_window):
    print('\rEpisode {}\tAverage Score: {:.2f}'.\
          format(i_episode, np.mean(scores_window)), end="")
    if i_episode % print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.\
              format(i_episode, np.mean(scores_window)))


def play_episode(env, agent, max_t, eps):
    score = 0.0
    state, _ = env.reset()
    #
    for _ in range(max_t):
        action = agent.act(state, eps)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    return score, state


def handle_restart(restart: bool, restart_name: str):
    """Handles restart"""
    if restart:
        if os.path.exists(restart_name):
            state = load(restart_name)
            return state, False
        else:
            # init from zerp
            return None, True
    else:
        if os.path.exists(restart_name):
            os.remove(restart_name)
        return None, True


def load(loadname: str):
    with open(f"{loadname}.pt", "rb") as fh:
        state = pickle.load(fh)
    return state


def save(savename, env, agent, i_episode, scores, scores_window, eps, save_every):
    if i_episode % save_every == 0:
        with open(f"{savename}.pt", "wb") as fh:
            pickle.dump(
                State(
                    env=env,
                    agent=agent,
                    i_episode=i_episode,
                    scores=scores,
                    scores_window=scores_window,
                    eps=eps
                )
                , fh)

        
def dqn(env_name,
        seed,
        algo,
        repeat,
        restart=True,
        scores_window_length=100,
        n_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        print_every=100,
        save_every=100):
    # init env
    restart_name = get_experiment_name(env_name, algo, repeat)
    state, init_from_zero_p = handle_restart(restart, restart_name)
    if init_from_zero_p:
        env, nA, nS = init_env(env_name, seed)
        episoide_begin = 1
        agent = Agent(algo, nS, nA, seed)
        scores, scores_window = [], deque(maxlen=scores_window_length)
        eps = eps_start
    else:
        env = state.env
        agent = state.agent
        episoide_begin = state.i_episode
        scores, scores_window = state.scores, state.scores_window
        eps = state.eps

    if episoide_begin >= n_episodes:
        return
    
    for i_episode in range(episoide_begin, n_episodes + 1):
        score, state = play_episode(env, agent, max_t, eps)
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print_info(i_episode, print_every, scores_window)
        save(restart_name, env, agent, i_episode, scores, scores_window, eps, save_every)



""" def dqn(env, seed, goal_fn,
        scores_window_length=100,
        n_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        print_every=100):

    # preparations
    nA, nS = init_env(env, seed)
    scores, scores_window = [], deque(maxlen=scores_window_length)
    agent = Agent(nS, nA, seed)
    eps = eps_start
    
    for i_episode in range(1, n_episodes + 1):
        score, state = play_episode(env, agent, max_t, eps)
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps * eps_decay)

        print_info(i_episode, print_every, scores_window)
        if goal_fn(scores_window, state , env):
            torch.save(agent.qnetwork_local.state_dict(),
                       f'{env.spec.id}_checkpoint.pth')
            break
    return scores
 """