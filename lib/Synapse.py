import numpy as np
from scipy.stats import truncnorm
from .Neuron import Neuron, NeuronType

class Synapse:

    def __init__(self,
                 pre: Neuron,
                 post: Neuron,
                 init_w: float = None,
                 scale: float = 0.1,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9):

        self.__pre = pre
        self.__post = post

        if init_w is None:
            self.__w = truncnorm.rvs(-2.0, 2.0, loc=0.0, scale=scale)
        else:
            self.__w = init_w
        
        self.__learning_rate = learning_rate
        self.__momentum = momentum

        self.__p_potentials_history = []
        self.__delta_update_factor = 0
    
    def reset(self):

        self.__p_potentials_history = []

    def __forward(self):

        # Pass forward from pre to post
        if np.not_equal(self.__pre.p_potential, 0).any():
            self.__post.send_p_potential(self.__pre.p_potential * self.__w,
                                         id=self.__pre.id)
            self.__p_potentials_history.append(self.__pre.p_potential)

    def __backward(self):

        # Pass backward from post to pre
        if np.not_equal(self.__post.e_potential, 0).any() and len(self.__p_potentials_history) > 0:
            
            e_potential = (self.__post.e_potential if self.__post.type != NeuronType.LOSS
                                                    else self.__post.e_potential[:, self.__pre.id])
            
            p = self.__p_potentials_history.pop(0)

            e_potential_to_send = e_potential * self.__w

            if self.__post.type != NeuronType.LOSS:
                allow_e_back = np.where(np.greater(p, 0), 1, 0)
                # allow_e_back = np.where(np.logical_and(np.greater(p, 0), np.less(p, 1)), 1, 0)
                e_potential_to_send = allow_e_back * e_potential_to_send

            self.__pre.send_e_potential(e_potential_to_send)

            # Update weight
            if self.__post.type != NeuronType.LOSS:
                update_factor = -self.__learning_rate * np.mean(p * e_potential)
                self.__delta_update_factor = self.__momentum * self.__delta_update_factor + update_factor
                self.__w = self.__w + self.__delta_update_factor

    def execute(self):

        self.__forward()
        self.__backward()
