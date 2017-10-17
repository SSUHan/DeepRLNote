import gym
from gym.envs.registration import register

register(
	id="FrozenLake-v3",
	entry_point="gym.envs.toy_text:FrozenLakeEnv",
	kwargs={'map_name':'4x4', 'is_slippery':False}
)


env = gym.make('FrozenLake-v3')
env.render()