import numpy as np
import gymnasium as gym
from memory import Memory
from model import BaseModel



class Agent():

    def __init__(self, env : gym.Env, memory : Memory, model : BaseModel, n_samples : int, eps : float, eps_dec : float, eps_min : float) -> None:
        self.env = env
        self.memory = memory
        self.model = model
        self.n_samples = n_samples
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min


    def act(self, state : np.ndarray, learning : bool) -> int:
        if learning:
            if np.random.rand() < self.eps:
                action = self.env.action_space.sample()
            else:
                pred = self.model.predict(state=state)
                action = np.argmax(pred)
            if self.eps > self.eps_min:
                self.eps *= self.eps_dec
                if self.eps_min > self.eps:
                    self.eps = self.eps_min
        else:
            pred = self.model.predict(state=state)
            action = np.argmax(pred)
        return action


    def learn(self) -> None:
        self.model.learn(mem=self.memory, n_samples=self.n_samples)


    def remember(self, state : np.ndarray, action : int, reward : float, new_state : np.ndarray, terminate : bool) -> None:
        self.memory.store(state=state, action=action, reward=reward, new_state=new_state, terminate=terminate)
