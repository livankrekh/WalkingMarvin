#!./venv/bin/python3

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense


model = model = Sequential([
  Dense(64, activation='relu', input_shape=(24,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])


env = gym.make('Marvin-v0')
env.reset()
for i_episode in range(1):
    observation = env.reset()
    for t in range(1):
        # env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
