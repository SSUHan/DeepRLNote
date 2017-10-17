import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

register(
	id="FrozenLake-v3",
	entry_point="gym.envs.toy_text:FrozenLakeEnv",
	kwargs={'map_name':'4x4', 'is_slippery':False}
)


env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Discount Factor
dis = 0.99

num_episodes = 2000

# Creat list to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
	state = env.reset()
	rAll = 0
	done = False

	# the Q-table learning algorithm
	while not done:
		action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
		
		# Get infomation from Env
		new_state, reward, done, _ = env.step(action)

		# Update Q-table with new knowledge using decay rate
		Q[state, action] = reward + dis*np.max(Q[new_state, :])

		rAll += reward
		state = new_state

	rList.append(rAll)
