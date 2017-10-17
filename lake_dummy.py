import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
	"""random argmax function"""
	m = np.amax(vector)
	indices = np.nonzero(vector == m)[0]
	return pr.choice(indices)

register(
	id="FrozenLake-v3",
	entry_point="gym.envs.toy_text:FrozenLakeEnv",
	kwargs={'map_name':'4x4', 'is_slippery':False}
)

env = gym.make('FrozenLake-v3')
env.render()

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000

# Creat list to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
	state = env.reset()
	rAll = 0
	done = False

	# the Q-table learning algorithm
	while not done:
		action = rargmax(Q[state, :])

		new_state, reward, done, _ = env.step(action)

		Q[state, action] = reward + np.max(Q[new_state, :])

		rAll += reward
		state = new_state

	rList.append(rAll)

print("Success Rate : ", str(sum(rList) / num_episodes))
print("Final Q-table Values")
print("Left Down Right, Up")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()
