from replay_buffer import ReplayBuffer
from algo import QTOpt

import gym
from gym import ObservationWrapper
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import trange


def device():
    GPU = True
    device_idx = 0
    if GPU:
        return torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


dev = device()


class CarRacing(gym.Wrapper):
    def __init__(self):
        env = gym.make('CarRacing-v0')
        shape = env.observation_space.shape

        self.crop = 15
        self.shape = (shape[0] - self.crop, shape[1])
        self.fs = 4
        self.frames = None

        env.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.fs, *self.shape))
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.observation(obs)
        self.frames = [obs] + self.frames[:-1]
        return np.asarray(self.frames), rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.observation(obs)
        self.frames = [np.zeros(self.shape, dtype=np.uint8)] * self.fs
        self.frames = [obs] + self.frames[:-1]
        return np.asarray(self.frames)

    def observation(self, obs):
        obs = obs[:-self.crop, :, :]
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        return obs


train_env = CarRacing()

action_space = train_env.action_space
obs_shape = train_env.observation_space.shape

replay_buffer_size = 30000
train_buffer_size = 10000


replay_buffer = ReplayBuffer(replay_buffer_size)
qt_opt = QTOpt(replay_buffer, obs_shape, action_space).to(dev)
assert qt_opt is not None


max_episodes = int(1e7)
batch_size = 100
max_steps = 200
episode_rewards = []



writer = SummaryWriter('logs/v3')

for i_episode in trange(0, max_episodes, batch_size):

    episode_reward = 0
    episode_actions = np.zeros(action_space.shape[0])

    state = train_env.reset()

    istep = 0
    done = False
    while not done and istep < max_steps:
        istep += 1
        state_batch = np.expand_dims(state, axis=0)
        action = qt_opt.cem_optimal_action(state_batch)
        next_state, reward, done, info = train_env.step(action)

        episode_reward += reward
        episode_actions += action

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

    if len(replay_buffer) > train_buffer_size:
        loss = qt_opt.update(batch_size)
        writer.add_scalar('loss', loss, i_episode)

        qt_opt.save_model('saved_model')

    writer.add_scalar('reward', episode_reward, i_episode)

    for i, a in enumerate(episode_actions):
        writer.add_scalar('actions_{}'.format(i), a, i_episode)

    print('Episode: {}  | Reward:  {}'.format(i_episode, episode_reward))
