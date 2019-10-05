#!./venv/bin/python3

import gym
import math
import numpy as np

env = gym.make('Marvin-v0')
env.reset()

input_size = 24
learning_rate = 0.1

layers_size = [input_size, 128, 32, 4]
nn = []
bias = []

def tanh(X):
	return (np.exp(X) - np.exp(X * -1)) / (np.exp(X) + np.exp(X * -1))

def tanh_derivative(X):
	return 1 - (self.tanh(X) ** 2)

def init_nn():
	global nn
	global bias

	for i in range(len(layers_size)-1):
		nn.append(np.random.randn(layers_size[i], layers_size[i+1]) * 0.1)
		bias.append(np.random.randn(layers_size[i+1], 1))

def nn_prop(observation):
	prev = observation

	for i in range(len(nn)):
		curr_layer = np.transpose(nn[i])
		curr_bias = bias[i]

		current_state = tanh(curr_layer.dot(prev) + curr_bias)
		prev = current_state

	return prev

def nn_backprop(observations, cum_reward):
	cumm_rewards = np.array(cum_reward)
	observations_arr = np.array(observations)

	loss_value = 1 - cumm_rewards

	losses = np.array([loss_value] * 4)
	

if __name__ == "__main__":
	init_nn()

	for i_episode in range(100):
		losses = []
		observation_arr = []
		cum_reward = 0

		observation = env.reset()
		for t in range(100):
			env.render()

			observation = np.array(observation).reshape((24,1))
			observation_arr.append(observation)

			action = nn_prop(observation).reshape((4))
			observation, reward, done, info = env.step(action)			

			cum_reward = reward - cum_reward
			losses.append(cum_reward)

			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break

		nn_backprop(observation_arr, losses)

	env.close()
