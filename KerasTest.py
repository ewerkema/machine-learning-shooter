import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


class Maze(object):
    def __init__(self, grid_size=10, random_locations=True):
        self.grid_size = grid_size
        self.grid = np.zeros(shape=(grid_size, grid_size))
        self.extra_reward = 0
        if random_locations:
            self.xplace = np.random.randint(0, self.grid_size, size=1)[0]
            self.yplace = np.random.randint(0, self.grid_size, size=1)[0]
            self.xgoal = np.random.randint(0, self.grid_size, size=1)[0]
            self.ygoal = np.random.randint(0, self.grid_size, size=1)[0]
        else:
            self.xplace = 1
            self.yplace = 1
            self.xgoal = grid_size - 1
            self.ygoal = grid_size - 1

        self.xlava = int(grid_size / 2)
        self.ylava = int(grid_size / 2)

        self.grid[self.xplace, self.yplace] = 5
        self.grid[self.xgoal, self.ygoal] = 10
        self.grid[self.xlava, self.ylava] = -10

    def _update_place(self, action):
        xplace = self.xplace
        yplace = self.yplace
        self.grid[self.xplace, self.yplace] = 0
        self.grid[self.xgoal, self.ygoal] = 10

        if action == 0:
            xplace -= 1
        elif action == 1:
            yplace -= 1
        elif action == 2:
            xplace += 1
        elif action == 3:
            yplace += 1

        if xplace < 0:
            xplace = 0
        if xplace >= self.grid_size - 1:
            xplace = self.grid_size - 1
        if yplace < 0:
            yplace = 0
        if yplace >= self.grid_size - 1:
            yplace = self.grid_size - 1
        self.xplace = xplace
        self.yplace = yplace
        if self.xplace == self.xgoal and self.yplace == self.ygoal:
            self.extra_reward += 1000
        if self.xplace == self.xlava and self.yplace == self.ylava:
            self.extra_reward -= 1000

        self.grid[self.xplace, self.yplace] = 5

    def get_reward(self):
        xreward = self.grid_size - abs(self.xplace - self.xgoal)
        yreward = self.grid_size - abs(self.yplace - self.ygoal)
        return xreward + yreward + self.extra_reward

    # action 0 = up, 1 = left, 2 = down, 3 = right
    def update_state(self, action):
        self._update_place(action)

    def print_grid(self):
        print self.grid
        print("X = ", maze.xplace)
        print("Y = ", maze.yplace)
        print("Reward = ", maze.get_reward(), "/20")

    def get_1d_grid(self):
        return self.grid.reshape((1, -1))


class Memory(object):
    def __init__(self, max_memory=100, discount=1):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=250):
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

    def __init__(self, grid_size, hidden_size=100, num_actions=4):
        # parameters
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.grid_size = grid_size
        self._init_model()

    def _init_model(self):
        # init model
        self.model = Sequential()
        self.model.add(Dense(self.hidden_size, input_shape=(self.grid_size ** 2,), activation='sigmoid'))
        #self.model.add(Dense(self.hidden_size, activation='sigmoid'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(sgd(lr=.2), "mse")
        self.memory = Memory()

    def predict_action(self, input_data, epsilon=.2):
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, self.num_actions, size=1)[0]
        else:
            q = self.model.predict(input_data)
            action = np.argmax(q[0])
        return action

    def get_new_state(self, input_data, action, reward, input_datap1):
        self.memory.remember([input_data, action, reward, input_datap1])
        inputs, targets = self.memory.get_batch(self.model)
        loss = self.model.train_on_batch(inputs, targets)
        return loss

grid_size = 10
epochs = 10
game_length = 30
random_locations = False

agent = SelfLearningAgent(grid_size)
rewards = np.zeros(epochs*game_length)

# Train games
for epoch in range(epochs):
    print epoch
    maze = Maze(grid_size=grid_size, random_locations=random_locations)
    input_data = maze.get_1d_grid()

    # Train Game
    for x in range(game_length):
        action = agent.predict_action(input_data)
        maze.update_state(action)
        input_datap1 = maze.get_1d_grid()
        reward = maze.get_reward()
        loss = agent.get_new_state(input_data, action, reward, input_datap1)
        input_data = input_datap1
        # print(loss)
        #TODO Loss is too big!
        if True: # x True for loss, False for Reward
            rewards[(epoch * game_length) + x] = loss
        else:
            rewards[(epoch*game_length)+x] = maze.get_reward()
        maze.print_grid()


x = range(0, epochs*game_length)
plt.plot(x, rewards)
plt.plot(x, np.zeros(epochs*game_length) + (grid_size*2))
# axes = plt.gca()
# axes.set_ylim([0, grid_size*3])

plt.show()


