from itertools import product
from dqn import dqn
from dqn_algorithms import DQN, DDQN, Dueling_DDQN
import numpy as np


SAVE_EVERY_N_EPISODES = 100
env_names = ["CartPole-v1", "LunarLander-v2", "MountainCar-v0"]
algos = [DQN, DDQN, Dueling_DDQN]
N_EPISODES_PER_ENV = {
    'CartPole-v1': 2000,
    'LunarLander-v2': 800,
    'MountainCar-v0': 800
}


# We run everything for a fixed number of episodes and don't care about task completion.
# This way we don't have to worry about the completion of different environements.
def main():
    # seed = 101
    repeats = range(1, 4)
    for env_name, algo, repeat in product(env_names, algos, repeats):
        seed = np.random.randint(1, 1000)
        dqn(env_name, seed, algo, repeat,
            n_episodes=N_EPISODES_PER_ENV[env_name],
            save_every=SAVE_EVERY_N_EPISODES)


if __name__ == "__main__":
    main()
