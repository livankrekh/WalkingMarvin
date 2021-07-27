#!./venv/bin/python3.7

import sys
import gym
import numpy as np

from utils.NNBackProp import *

env = gym.make('Marvin-v0')
env.reset()

if __name__ == "__main__":
	
	nn = NNBackProp()
	nn.init_nn()

	if "-r" in sys.argv:
		nn.load_weights("models/model_1000.pickle")

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

			action = nn.predict(observation).reshape((4))
			observation, reward, done, info = env.step(action)

			cum_reward = reward - cum_reward

			losses.append(cum_reward)

			if done:
				print("Episode finished after {} timesteps".format(i_episode+1))
				break

		nn.nn_backprop(observation_arr, losses)

	done = False
	observation = env.reset()

	env.close()