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

    def sample(self, batch_size):
        choices = np.array(self.buffer)
        idx = random.choice(len(choices), batch_size)
        batch = choices[idx]
        state, action, reward, next_state, done = map(np.concatenate, zip(
            *batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
