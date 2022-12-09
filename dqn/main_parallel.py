from itertools import product, repeat
from dqn import dqn
from dqn_algorithms import DQN, DDQN, Dueling_DDQN
import numpy as np
import multiprocessing as mp


def dqn_aux(env_name, algo, exp_repeat, seed, n_episodes, save_every):
    dqn(env_name, seed, algo, exp_repeat, n_episodes=n_episodes, save_every=save_every)


def main():
    SAVE_EVERY_N_EPISODES = 100
    env_names = ["CartPole-v1", "LunarLander-v2", "MountainCar-v0"]
    N_EPISODES_PER_ENV = {
        'CartPole-v1': 2000,
        'LunarLander-v2': 800,
        'MountainCar-v0': 400
    }
    algos = [DQN, DDQN, Dueling_DDQN]

    args = list(product(env_names, algos, range(1, 4)))
    env_names = [arg[0] for arg in args]
    algos = [arg[1] for arg in args]
    exp_repeats = [arg[2] for arg in args]
    seeds = [np.random.randint(1, 1000) for _ in range(len(args))]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        print(mp.cpu_count())
        pool.starmap(
            dqn_aux,
            zip(
                env_names,
                algos,
                exp_repeats,
                seeds,
                list(repeat(N_EPISODES_PER_ENV["CartPole-v1"], 9)) + \
                list(repeat(N_EPISODES_PER_ENV["LunarLander-v2"], 9)) + \
                list(repeat(N_EPISODES_PER_ENV["MountainCar-v0"], 9)),
                repeat(SAVE_EVERY_N_EPISODES, len(args)),
            )
        )


if __name__ == "__main__":
    main()
