from collections import deque
from agent import Agent

import torch
import numpy as np


def init_env(env, seed):
    env.reset(seed=seed)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    return nA, nS


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


def dqn(env, seed, goal_fn,
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
