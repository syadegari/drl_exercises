from dqn import dqn



N_EPISODES = 200

def dqn():
    if restart:
        # look for restart files
        # load the restarts
        episoide_begin = ... # from saved file
        # handle the case where episoide_begin == n_episodes
        ...

    if not restart:
        # fresh start:
        #   look for any restart files
        #   delete them
        scores, scores_window = [], deque(maxlen=scores_window_length)
        agent = Agent(nS, nA, seed)
        eps = eps_start
        episoide_begin = 1

    # init env
    env.reset(seed=seed)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    agent = Agent(nS, nA, seed)

    for i_episode in range(episoide_begin, n_episodes + 1):
        
    


def main():
    for env in envs:
        for algo in algos:
            for repeat in repeats:
                dqn(env, algo, repeat, n_episodes=N_EPISODES, restart=True)

if __name__ == '__main__':
    main()
