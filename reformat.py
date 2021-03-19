#!./venv/bin/python3

import gym
import numpy as np

env = gym.make('Marvin-v0')
env.reset()

# np.random.seed(67)
np.random.seed(12334)

input_size = 24
learning_rate = 0.000001

layers_size = [input_size, 32, 4]
nn = []
bias = []

def tanh(X):
	return (np.exp(X) - np.exp(X * -1)) / (np.exp(X) + np.exp(X * -1))

def tanh_derivative(X):
	return 1 - (tanh(X) ** 2)

def relu(X):
	return np.maximum(X, 0)

def relu_derivative(X):
	X[X <= 0] = 0
	X[X > 0] = 1

	return X

activtions = [relu, tanh]
derivatives = [relu_derivative, tanh_derivative]

def init_nn():
	global nn
	global bias

	for i in range(len(layers_size)-1):
		nn.append(np.random.randn(layers_size[i], layers_size[i+1]))
		bias.append(np.random.randn(layers_size[i+1], 1))

def nn_prop(observation):
	prev_dim = 1 if len(observation.shape) < 2 else observation.shape[1]
	prev = observation.reshape( (observation.shape[0], prev_dim) )

	for i in range(len(nn)):
		curr_layer = np.transpose(nn[i])
		curr_bias = bias[i]

		current_state = activtions[i](curr_layer.dot(prev) + curr_bias)
		prev = current_state

	return prev

def get_nn_prop_history(observation):
	prev_dim = 1 if len(observation.shape) < 2 else observation.shape[1]
	prev = observation.reshape( (observation.shape[0], prev_dim) )
	history = []

	for i in range(len(nn)):
		curr_layer = np.transpose(nn[i])
		curr_bias = bias[i]

		current_state = activtions[i](curr_layer.dot(prev) + curr_bias)
		prev = current_state
		history.append(current_state)

	return history

def nn_backprop_one(observation, cum_reward):
	history = get_nn_prop_history(observation)

	# loss_value = 1 - cum_reward

	losses = np.transpose(np.array([cum_reward] * 4))
	o_0 = losses * tanh_derivative(history[-2].T.dot(nn[-1]) + bias[-1].reshape((bias[-1].shape[0],)))
	o_0 = o_0[0]

	o_1 = []

	for i in range(layers_size[-2]):
		o_1_tmp = 0
		for j in range(layers_size[-1]):
			delta = learning_rate * o_0[j] * history[-2][i]
			o_1_tmp += nn[-1][i][j] * o_0[j]
			nn[-1][i][j] += delta

		o_1.append(o_1_tmp)

	o_1 = np.array(o_1)
	o_1 = o_1 * relu_derivative(observation.T.dot(nn[-2]) + bias[-2].reshape((bias[-2].shape[0],)))

	for i in range(layers_size[-3]):
		for j in range(layers_size[-2]):
			delta = learning_rate * o_1[j] * observation[i]
			nn[-2][i][j] += delta

def nn_backprop(observations, cum_rewards):
	for i in range(len(observations)):
		nn_backprop_one(observations[i], cum_rewards[i])

if __name__ == "__main__":
	init_nn()

	for i_episode in range(50):
		losses = []
		observation_arr = []
		cum_reward = 0
		done = False

		print("Epoch #", i_episode)

		observation = env.reset()
		while not done:
			env.render()

			observation = np.array(observation).reshape((24,))
			observation_arr.append(observation)

			action = nn_prop(observation).reshape((4))
			observation, reward, done, info = env.step(action)

			cum_reward = reward - cum_reward
			# nn_backprop_one(observation, cum_reward)

			losses.append(cum_reward)

			if done:
				# print("Episode finished after {} timesteps".format(t+1))
				break

		nn_backprop(observation_arr, losses)

	done = False
	observation = env.reset()

	while not done:
		env.render()
		observation = np.array(observation).reshape((24,1))
		observation_arr.append(observation)

		action = nn_prop(observation).reshape((4))
		observation, reward, done, info = env.step(action)

	env.close()
