import numpy as np
import gym
import pickle
import matplotlib.pyplot as plt
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--learning", default=True)
parser.add_argument("--render" , default=False)
parser.add_argument("--resume_step", default=0, type=int)
args = parser.parse_args()

MODEL_SAVE_PATH = os.path.join(".", 'pong_models')
if not os.path.exists(MODEL_SAVE_PATH):
	os.mkdir(MODEL_SAVE_PATH)
	print("{} generated..".format(MODEL_SAVE_PATH))

is_learning  = args.learning
if args.resume_step == 0 :
	is_resume, chp_step = False, 0
else:
	is_resume, chp_step = True, args.resume_step	# using checkpoint?

is_render = args.render

# Hyperparameters
HIDDEN_NEURONS = 200	# number of hidden layer neurons
batch_size = 10
lr = 1e-4				# learning rate
discount_rate = 0.99 	# discount factor for reward
decay_rate = 0.99		# decay factor for RMSProp
model_save_step = 500	# model save step

INPUT_DIM = 80 * 80 	# input dimensionality: 80 * 80 grid

if is_resume:
	model = pickle.load(open(os.path.join(MODEL_SAVE_PATH, 'save_ep{}.p'.format(chp_step)), 'rb'))
else:
	model = {}
	model['W1'] = np.random.randn(HIDDEN_NEURONS, INPUT_DIM) / np.sqrt(INPUT_DIM) # "xavier" initialization
	model['W2'] = np.random.randn(HIDDEN_NEURONS) / np.sqrt(HIDDEN_NEURONS)

grad_buffer = {k:np.zeros_like(v) for k, v in model.items()} # update buffers that add up gradients over a batch
rmsprop_cache = {k:np.zeros_like(v) for k, v in model.items()} 

def _sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def preprocessing(image):
	""" origin image shape : 210*160*3 """
	image = image[35:195] # playground field cropping, shape : 160 * 160 * 3
	image = image[::2, ::2, 0] # downsample by 2*2, shape : 80 * 80 * 1
	image[image == 144] = 0.0 # erase background type 1 (hard coding)
	image[image == 109] = 0.0 # erase background type 2 (hard coding)
	image[image != 0] = 1.0 # paddles, ball mark 1
	return image.ravel()

def policy_forward(x):
	h = np.dot(model['W1'], x)
	h[h<0] = 0 # ReLU effect
	logp = np.dot(model['W2'], h)
	p = _sigmoid(logp)
	return p, h

def discount_reward(r):
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		if r[t] != 0:
			running_add = 0
		running_add = running_add*discount_rate + r[t]
		discounted_r[t] = running_add
	return discounted_r

def policy_backward(ep_x, ep_h, ep_dlogp):
	"""
		ep_h is array of intermediate hidden states
	"""
	dW2 = np.dot(ep_h.T, ep_dlogp).ravel()
	dh = np.outer(ep_dlogp, model['W2']) # outer : 외적
	dh[ep_h <= 0] = 0 # backpro prelu
	dW1 = np.dot(dh.T, ep_x)
	return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset()		# return output image (210, 160, 3)

prev_x = None # using for computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = chp_step


if not is_learning:
	plt.imshow(observation)
	plt.show()

	plt.imshow(preprocessing(observation), cmap='gray')
	plt.show()

	while True:
		env.render()
		observation, reward, done, info = env.step(random.randint(1,3))
		if reward != 0.0:
			print(reward)
		
		if done:
			print("play done..")
			break
else:
	while True:
		if is_render:
			env.render()

		current_x = preprocessing(observation)
		if prev_x is not None:
			x = current_x - prev_x
		else:
			x = np.zeros(INPUT_DIM)
		
		prev_x = current_x

		aprob, h = policy_forward(x)
		if np.random.uniform() < aprob:
			action = 2
		else:
			action = 3

		# record various intermediates (needed later for backprop)
		xs.append(x) # record observations
		hs.append(h) # record hidden state 
		y = 1 if action == 2 else 0
		dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken

		# step the environment and get new measurements
		observation, reward, done, info = env.step(action)
		reward_sum += reward

		drs.append(reward) # record reward

		if done: 
			# an episoid finished
			episode_number += 1

			# stack together all inputs, hidden states, action gradients, and rewards for this episode.
			ep_x = np.vstack(xs)
			ep_h = np.vstack(hs)
			ep_dlogp = np.vstack(dlogps)
			ep_r = np.vstack(drs)
			xs, hs, dlogps, drs = [], [], [], [] # reset memory

			# compute the discounted reward backwards through time
			discounted_epr = discount_reward(ep_r)
			# standardize the rewards to be unit normal
			discounted_epr -= np.mean(discounted_epr)
			discounted_epr /= np.std(discounted_epr)

			ep_dlogp *= discounted_epr # modulate the gradient with advantage
			grad = policy_backward(ep_x, ep_h, ep_dlogp)
			for k in model:
				grad_buffer[k] += grad[k]

			# perform rmsprop parameter update every batch_size episodes
			if episode_number % batch_size == 0:
				for k, v in model.items():
					g = grad_buffer[k]
					rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
					model[k] += lr * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
					grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

			running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
			print('Reset env. Episode reward total was : {}. running mean : {}'.format(reward_sum, running_reward))
			if episode_number % model_save_step == 0:
				pickle.dump(model, open(os.path.join(MODEL_SAVE_PATH, 'save_ep{}.p'.format(episode_number)), 'wb'))

			reward_sum = 0
			observation = env.reset() # reset env
			prev_x = None

		if reward != 0:
			print("Ep : {} game finished, reward : {} {}".format(episode_number, reward, '' if reward == -1 else ' !!!!!'))








