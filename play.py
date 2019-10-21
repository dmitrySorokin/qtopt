import gym
from gym.utils.play import play




env = gym.make('CarRacing-v0')

a = env.action_space
print(a.low, a.high)
print(env.observation_space)

obs = env.reset()
print(obs.shape)

print(dir(play))
play(env, zoom=3, keys_to_action={'w': (1, 1, 1)})