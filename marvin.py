import gym
import numpy as np

from NNBackProp import *

env = gym.make('Marvin-v0')
env.reset()

if __name__ == "__main__":
	nn = NNBackProp()

	nn.init_nn()

	for i_episode in range(50):
		actions = []
		observations = []
		rewards = []
		done = False

		print("Epoch #", i_episode)

		observation = env.reset()
		while not done:
			env.render()

			observation = np.array(observation).reshape((24,))
			observations.append(observation)

			action = nn.predict(observation).reshape((4))
			observation, reward, done, info = env.step(action)

			rewards.append(reward)
			actions.append(action)

			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break

		nn.backprop(observations, actions, rewards)

	done = False
	observation = env.reset()

	while not done:
		env.render()
		observation = np.array(observation).reshape((24,1))
		observation_arr.append(observation)

		action = nn.nn_prop(observation).reshape((4))
		observation, reward, done, info = env.step(action)

	env.close()