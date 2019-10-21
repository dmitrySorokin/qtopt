import numpy as np
from numpy import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def _encode_sample(self, idxes):
        obses, actions, rewards, next_obses, dones = [], [], [], [], []

        for i in idxes:
            (obs, action, reward, next_obs, done) = self.buffer[i]
            obses.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_obses.append(next_obs)
            dones.append(done)
        return \
            np.asanyarray(obses), \
            np.asanyarray(actions), \
            np.asanyarray(rewards), \
            np.asanyarray(next_obses), \
            np.asanyarray(dones)


    def sample(self, batch_size):
        idx = random.choice(len(self.buffer), batch_size)
        return self._encode_sample(idx)

    def __len__(self):
        return len(self.buffer)
