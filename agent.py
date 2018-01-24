import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import sgd


class Memory(object):
	def __init__(self, max_memory=100, discount=.99):
		self.max_memory = max_memory
		self.memory = list()
		self.discount = discount

	def remember(self, states):
		# memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
		self.memory.append([states])
		if len(self.memory) > self.max_memory:
			del self.memory[0]

	def get_batch(self, model, batch_size=50):
		len_memory = len(self.memory)
		num_actions = model.output_shape[-1]
		env_dim = self.memory[0][0][0].shape[1]
		inputs = np.zeros((min(len_memory, batch_size), env_dim))
		targets = np.zeros((inputs.shape[0], num_actions))
		for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
			state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

			inputs[i:i+1] = state_t
			# There should be no target values for actions not taken.
			# Thou shalt not correct actions not taken #deep
			targets[i] = model.predict(state_t)[0]
			Q_sa = np.max(model.predict(state_tp1)[0])
			# reward_t + gamma * max_a' Q(s', a')
			targets[i, action_t] = reward_t + self.discount * Q_sa
		return inputs, targets


class SelfLearningAgent(object):

	def __init__(self, input_size, hidden_size=150, num_actions=5):
		# parameters
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_actions = num_actions
		self._init_model()

	def _init_model(self):
		# init model
		self.model = Sequential()
		self.model.add(Dense(self.hidden_size, input_shape=(self.input_size, ), activation='sigmoid'))
		self.model.add(Dense(self.num_actions, activation='linear'))
		self.model.compile(optimizer=sgd(lr=1e-03), loss="mse")
		self.memory = Memory()

	def predict_action(self, input_data, epsilon=.1):
		if np.random.rand() <= epsilon:
			action = np.random.randint(0, self.num_actions, size=1)[0]
		else:
			q = self.model.predict(input_data, batch_size=self.input_size)[0]
			action = np.argmax(q)
		return action
		#q_values = self.model.predict(input_data)[0]
		# totalQ = sum(q_values)
		# rand = np.random.uniform(0, totalQ)
		# q = 0
		# total = 0
		# for i in range(len(q_values)):
		# 	total += q_values[i]
		# 	if rand <= total:
		# 		q = i
		# 		break
		# return q

	def get_new_state(self, input_data, action, reward, input_datap1):
		self.memory.remember([input_data, action, reward, input_datap1])
		inputs, targets = self.memory.get_batch(self.model)
		loss = self.model.train_on_batch(inputs, targets)
		return loss
