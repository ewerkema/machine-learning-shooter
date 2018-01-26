import numpy as np

import config


class AbstractMemory(object):
    """Q Learning memory class for remembering states
    of the implemented agent"""
    def clear(self):
        raise NotImplementedError("Class %s doesn't implement clear()" % self.__class__.__name__)

    def remember(self, states):
        raise NotImplementedError("Class %s doesn't implement remember(states)" % self.__class__.__name__)

    def get_batch(self, model, batch_size=50):
        raise NotImplementedError("Class %s doesn't implement get_batch(model, batch_size=50)" % self.__class__.__name__)


class AbstractAgent(object):
    """A self-learning agent that is implemented by a certain
    keras model. This class represents an interface for an agent"""
    def __init__(self):
        self.num_actions = len(config.actions)
        self.q = np.zeros(self.num_actions)

    def _init_model(self):
        raise NotImplementedError("Class %s doesn't implement _init_model()" % self.__class__.__name__)

    def predict_action(self, input_data, epsilon=.1):
        raise NotImplementedError("Class %s doesn't implement predict_action(input_data, epsilon=.1)" % self.__class__.__name__)

    def get_new_state(self, input_data, action, reward, input_datap1):
        raise NotImplementedError("Class %s doesn't implement get_new_state(input_data, action, reward, "
                                  "input_datap1):" % self.__class__.__name__)

    def get_q_values(self):
        return self.q
