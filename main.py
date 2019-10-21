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


class CarRacing(ObservationWrapper):
    def __init__(self):
        env = gym.make('CarRacing-v0')
        shape = env.observation_space.shape
        env.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[2], shape[0], shape[1]))
        super().__init__(env)

    def observation(self, obs):
        return obs.transpose(2, 0, 1)


train_env = CarRacing()

action_space = train_env.action_space
obs_shape = train_env.observation_space.shape

replay_buffer_size = 300


replay_buffer = ReplayBuffer(replay_buffer_size)
qt_opt = QTOpt(replay_buffer, obs_shape, action_space).to(dev)
assert qt_opt is not None


max_episodes = int(1e7)
batch_size = 100
max_steps = 200
episode_rewards = []



writer = SummaryWriter('logs/v1')

for i_episode in trange(0, max_episodes, batch_size):

    episode_reward = 0
    episode_dist = 0
    episode_angle = 0
    episode_actions = np.zeros(action_space.shape[0])

    state = train_env.reset()
    state = np.expand_dims(state, axis=0)
    print(state.shape)

    for step in range(max_steps):
        # action = qt_opt.policy.act(state)
        action = qt_opt.cem_optimal_action(state)
        next_state, reward, done, info = train_env.step(action)
        next_state = np.expand_dims(next_state, axis=0)

        episode_reward += reward
        episode_actions += action


        replay_buffer.push(state, [action], [reward], next_state, [done])
        state = next_state

    if len(replay_buffer) > batch_size:
        loss = qt_opt.update(batch_size)
        writer.add_scalar('loss', loss, i_episode)

        qt_opt.save_model('saved_model')

    writer.add_scalar('reward', episode_reward, i_episode)
    writer.add_scalar('dist', episode_dist, i_episode)
    writer.add_scalar('angle', episode_angle, i_episode)

    for i, a in enumerate(episode_actions):
        writer.add_scalar('actions_{}'.format(i), a, i_episode)

    print('Episode: {}  | Reward:  {}'.format(i_episode, episode_reward))