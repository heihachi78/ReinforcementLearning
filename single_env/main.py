import gymnasium as gym
import tensorflow as tf
from simulator import Simulator
from memory import Memory
from model import ModelNaive, ModelActorCritic
from agent import Agent



def disable_gpu(disable : bool = False) -> str:
    physical_gpu_devices = tf.config.list_physical_devices('GPU')
    print(f'physical_gpu_devices {physical_gpu_devices}')
    physical_cpu_devices = tf.config.list_physical_devices('CPU')
    print(f'physical_cpu_devices {physical_cpu_devices}')
    device = 'GPU'
    if disable:
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            print(f'visible_devices {visible_devices}')
            for device in visible_devices:
                assert device.device_type != 'GPU'
            device = 'CPU'
        except:
            print('cannot disable gpu')
    return device


LEARNING = True
DISABLE_GPU = False
RENDER = False
EPISODES = 1_000
MAX_MEMORY = 100_000
ENV_NAME = 'LunarLander-v2'
LEARNING_RATE = 5e-4
NEURONS = [256, 128]
VERBOSE = 0
N_LEARN_SAMPLES = 64
GAMMA = 0.99
WEIGHT_SAVE_INTERVAL = 25
WEIGHT_REFRESH_INTERVAL = 4
HISTORY_SAVE_INTERVAL = 10
EPS = 1
EPS_DEC = 1-10e-4
EPS_MIN = 0.1



def setup(device : str):
    if RENDER:
        env = gym.make(ENV_NAME, render_mode="human")
    else:
        env = gym.make(ENV_NAME)

    mem = Memory(max_memory=MAX_MEMORY,
                state_shape=env.observation_space.shape,
                n_actions=env.action_space.n)

    nn = ModelActorCritic(n_input=env.observation_space.shape[0],
            n_output=env.action_space.n,
            n_neurons=NEURONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            save_file=ENV_NAME,
            verbose=VERBOSE,
            device=device)

    agent = Agent(env=env,
                memory=mem,
                model=nn,
                n_samples=N_LEARN_SAMPLES,
                eps=EPS,
                eps_dec=EPS_DEC,
                eps_min=EPS_MIN)

    sim = Simulator(agent=agent,
                    env=env,
                    episodes=EPISODES,
                    weight_save_interval=WEIGHT_SAVE_INTERVAL,
                    weight_refresh_interval=WEIGHT_REFRESH_INTERVAL,
                    history_save_interval=HISTORY_SAVE_INTERVAL,
                    device=device)
    
    return sim



if __name__ == '__main__':
    device = disable_gpu(DISABLE_GPU)
    sim = setup(device=device)
    sim.sim_loop(learning=LEARNING)
