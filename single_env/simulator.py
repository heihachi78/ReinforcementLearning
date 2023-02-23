import gymnasium as gym
import time
import numpy as np
from agent import Agent



class Simulator():

    def __init__(self, episodes : int, env : gym.Env, agent : Agent, weight_save_interval : int, weight_refresh_interval : int, history_save_interval : int, device : str) -> None:
        self.episodes = episodes
        self.env = env
        self.agent = agent
        self.weight_save_interval = weight_save_interval
        self.weight_refresh_interval = weight_refresh_interval
        self.history_save_interval = history_save_interval
        self.score_history = []
        self.device = device


    def _debug_log(self, episode : int, reward : float, step : int, duration : float) -> None:
        print(f'ep={episode} rew={int(reward)} avg100={int(np.mean(self.score_history[-100:]))} avg50={int(np.mean(self.score_history[-50:]))} st={step} mc={self.agent.memory.count} dur={duration:.3f} eps={self.agent.eps:.3f}', end = ' ')


    def sim_loop(self, learning : bool) -> None:
        self.agent.model.load()
        
        for episode_n in range(self.episodes):
            observation, _ = self.env.reset()
            terminated = False
            truncated = False
            collected_reward = 0
            step = 0
            t_start = time.perf_counter()

            while not(terminated or truncated):
                step += 1
                old_observation = observation
                action = self.agent.act(state=old_observation, learning=learning)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                self.agent.remember(state=old_observation, action=action, reward=reward, new_state=observation, terminate=(terminated or truncated))
                if learning:
                    self.agent.learn()
                collected_reward += reward

                if terminated or truncated:
                    break

            self.score_history.append(collected_reward)

            t_end = time.perf_counter()
            t_duration = t_end - t_start

            self._debug_log(episode=episode_n, reward=collected_reward, step=step, duration=t_duration)

            if learning:
                if not(episode_n % self.weight_refresh_interval):
                    self.agent.model.refresh_weights()
                    print('(wr)', end=' ')

                if not(episode_n % self.weight_save_interval):
                    self.agent.model.save()
                    print('(ws)', end=' ')

                if not(episode_n % self.history_save_interval):
                    np.save(f'hist_{self.env.unwrapped.spec.id}_{self.agent.model.__class__.__name__}_{self.device}', np.array(self.score_history, dtype=np.float32))
                    print('(hs)', end=' ')

            print('')

        self.env.close()
