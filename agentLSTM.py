import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import sgd
import config
from agent import AbstractAgent, AbstractMemory

TIMESTEPS = 20


class Memory(AbstractMemory):
    def __init__(self, max_memory=TIMESTEPS*3, discount=.99):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def clear(self):
        self.memory = list()

    def remember(self, states):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_time_seq(self, idx):
        if idx == 0:
            idx = len(self.memory) - TIMESTEPS
        env_dim = self.memory[0][0][0].shape[1]
        time_seq = np.zeros([TIMESTEPS, env_dim])
        for j in range(TIMESTEPS):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx + j][0]
            time_seq[j] = state_t
        time_seq = np.expand_dims(time_seq, 0)
        return time_seq

    def get_batch(self, model, batch_size=1):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((batch_size, TIMESTEPS, env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory - TIMESTEPS, size=inputs.shape[0])):
            _, action_t, reward_t, statep1 = self.memory[idx + TIMESTEPS][0]
            time_seq = self.get_time_seq(idx)
            inputs[i] = time_seq
            targets[i] = model.predict(time_seq)[0]
            # Delete the first state and add the next state
            time_seqp1 = np.delete(time_seq, 0, 1)
            time_seqp1 = np.append(time_seqp1, [statep1], 1)
            Q_sa = np.max(model.predict(time_seqp1)[0])
            # reward_t + gamma * max_a' Q(s', a')
            targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


class Agent(AbstractAgent):

    def __init__(self, input_size, hidden_size=150):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._init_model()

    def _init_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_size, return_sequences=True, input_shape=(TIMESTEPS, self.input_size)))
        self.model.add(LSTM(self.hidden_size, return_sequences=False))
        self.model.add(Dense(self.num_actions, activation='linear'))
        self.model.compile(optimizer=sgd(lr=1e-03), loss="mse")
        self.memory = Memory()

    def predict_action(self, input_data, epsilon=.1):
        if np.random.rand() <= epsilon or len(self.memory.memory) < TIMESTEPS:
            action = np.random.randint(0, self.num_actions, size=1)[0]
        else:
            input_data = self.memory.get_time_seq(0)
            self.q = self.model.predict(input_data, batch_size=self.input_size)[0]
            action = np.argmax(self.q)
        return action

    def get_new_state(self, input_data, action, reward, input_datap1):
        self.memory.remember([input_data, action, reward, input_datap1])
        loss = 0
        if len(self.memory.memory) > TIMESTEPS:
            inputs, targets = self.memory.get_batch(self.model)
            loss = self.model.train_on_batch(inputs, targets)

        return loss

