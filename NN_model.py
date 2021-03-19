import numpy as np
import pickle

class Model:
    def __init__(self):
        np.random.seed(12334)

        self.input_size = 24
        self.learning_rate = 0.000001

        self.layers_size = [self.input_size, 32, 4]
        self.nn = []
        self.bias = []

        self.activtions = []
        self.derivatives = []

"""
Activation functions
___________________________________________________________________________________________-

"""

    def _tanh(self, X):
        return (np.exp(X) - np.exp(X * -1)) / (np.exp(X) + np.exp(X * -1))

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(X * -1))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _relu(self, X):
        return np.maximum(X, 0)

    def _leaky_relu(self, X):
        return np.where(X >= 0, X, X * 0.01)

    def _pelu(self, X, coff):
        return np.where(X >= 0, X, X * coff)

    def _elu(self, X, coff):
        return np.where(X >= 0, X, (np.exp(X) - 1) * coff)

    def _linear(self, X, coff=1.0):
        return X * coff

    def _arctanh(self, X):
        return 1 / np.tan(X)

    def _sin(self, X):
        return np.sin(X)

    def _gaussian(self, X):
        return np.exp(X**2 * -1)

"""
Derivatives
___________________________________________________________________________________________-

"""

    def _tanh_derivative(self, X):
        return 1 - (self.tanh(X) ** 2)

    def _sigmoid_derivative(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def _softmax_derivative(self, X):
        
        def _point_derivative(o, i, j):
            if i == j:
                return None
            return None

    def _reLu_derivative(self, X):
        return np.where(X >= 0, 1, 0)

    def _leaky_relu_derivative(self, X):
        return np.where(X >= 0, 1, 0.01)

    def _pelu_derivative(self, X, coff):
        return np.where(X >= 0, 1, coff)

    def _elu_derivative(self, X, coff):
        return np.where(X >= 0, 1, self._elu(X, coff) + coff)

    def _linear_derivative(self, X):
        return np.ones(X.shape)

    def _arctanh_derivative(self, X):
        return 1 / (X ** 2 + 1)

    def _sin_derivative(self, X):
        return np.cos(X)

    def _gaussian_derivative(self, X):
        return np.exp(X**2 * -1) * -2 * X

"""___________________________________________________________________________________________-

"""

    def init_nn(self):
        self.activtions = [self.relu, self.tanh]
        self.derivatives = [self.relu_derivative, self.tanh_derivative]

        for i in range(len(self.layers_size)-1):
            self.nn.append(np.random.randn(self.layers_size[i], self.layers_size[i+1]))
            self.bias.append(np.random.randn(self.layers_size[i+1], 1))

    def get_nn_prop_history(self, X):
        prev_dim = 1 if len(X.shape) < 2 else X.shape[1]
        prev = X.reshape( (X.shape[0], prev_dim) )
        history = []

        for i in range(len(self.nn)):
            curr_layer = np.transpose(self.nn[i])
            curr_bias = self.bias[i]

            current_state = self.activtions[i](curr_layer.dot(prev) + curr_bias)
            prev = current_state
            history.append(current_state)

        return history

    def backprop_one(self, observation, cum_reward):
        pass

    def backprop(self, observations, actions, rewards):
        cum_rewards = []
        cum_rew = 0

        for reward in rewards:
            cum_rew = reward - cum_rew
            cum_rewards.append(cum_rew)

        for i in range(len(observations)):
            self.backprop_one(observations[i], cum_rewards[i])

    def predict(self, X):
        prev_dim = 1 if len(X.shape) < 2 else X.shape[1]
        prev = X.reshape( (X.shape[0], prev_dim) )

        for i in range(len(self.nn)):
            curr_layer = np.transpose(self.nn[i])
            curr_bias = self.bias[i]

            current_state = self.activtions[i](curr_layer.dot(prev) + curr_bias)
            prev = current_state

        return prev

    def save_weights(self, file):
        model = [self.nn, self.bias]
        pickle.dump(model, open(file, 'wb'))

    def load_weights(self, file):
        model = pickle.load(open(file, 'rb'))

        self.nn = model[0]
        self.bias = model[1]


