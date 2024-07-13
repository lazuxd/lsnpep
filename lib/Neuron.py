import numpy as np
from typing import Callable
from enum import Enum


class NeuronType(Enum):
    INPUT  = 1
    INNER  = 2
    OUTPUT = 3
    LOSS   = 4
    BIAS   = 5


class Neuron:

    def __init__(self,
                 type: NeuronType,
                 id: int = None,
                 num_outputs: int = None):

        self.__time = 0

        self.__type = type

        # potential is a vector in R^b, where b is the batch size
        self.__p_potential = None # .reset() should be used
        self.__e_potential = None # .reset() should be used

        self.__pending_p_potentials = []
        self.__pending_e_potentials = []

        if self.type == NeuronType.OUTPUT:
            if id is None:
                raise ValueError("The output neurons should have an id")
            self.__id = id

        elif self.type == NeuronType.LOSS:
            if num_outputs is None:
                raise ValueError("The loss neuron should have the number of outputs")
            self.__num_outputs = num_outputs
    
    def __repr__(self):
        return f'Neuron(type={self.__type}) with: time={self.__time}'

    @property
    def type(self) -> NeuronType:
        return self.__type

    @property
    def id(self) -> int:
        if self.type == NeuronType.OUTPUT:
            return self.__id
        else:
            return None
    
    @property
    def p_potential(self) -> np.ndarray:
        if self.type == NeuronType.BIAS:
            return np.ones_like(self.__p_potential)
        return self.__p_potential
    
    @property
    def e_potential(self) -> np.ndarray:
        return self.__e_potential
    
    def reset(self, p: np.ndarray, d_loss_fn: Callable = None):

        if self.type == NeuronType.LOSS:

            self.__d_loss_fn = d_loss_fn

            self.__pending_p_potentials = []
            self.__pending_p_potentials_ids = []
            self.__pending_e_potentials = []
        
        else:

            self.__pending_p_potentials = [p]
            self.__pending_e_potentials = [np.zeros_like(p)]
        
        self.__p_potential = np.zeros_like(p)
        self.__e_potential = np.zeros_like(p)
    
    def send_p_potential(self, p: np.ndarray, id: int = None):
        self.__pending_p_potentials.append(p)
        if self.type == NeuronType.LOSS:
            if id is None:
                raise ValueError('To Loss Neuron should be send also the id from the output neuron')
            self.__pending_p_potentials_ids.append(id)
    
    def send_e_potential(self, e: np.ndarray):
        self.__pending_e_potentials.append(e)
    
    def execute(self):

        self.__update_p_potential()
        self.__update_e_potential()

        self.__time = self.__time + 1

    def __update_p_potential(self):

        new_p_potential = np.zeros_like(self.p_potential)

        threshold_num = 1 if (self.type == NeuronType.INNER or self.type == NeuronType.OUTPUT) else 0

        if len(self.__pending_p_potentials) > threshold_num:

            if self.__type == NeuronType.LOSS:

                if self.__d_loss_fn is not None:

                    # As a side effect, also update e potential when reaching Loss Neuron

                    received_p_potentials = np.zeros((self.p_potential.shape[0], self.__num_outputs))
                    for i in range(len(self.__pending_p_potentials)):
                        neuron_id = self.__pending_p_potentials_ids[i]
                        val = self.__pending_p_potentials[i]
                        received_p_potentials[:, neuron_id] = val
                    
                    self.__e_potential = self.__d_loss_fn(received_p_potentials)

                    self.__pending_p_potentials_ids = []

            else:

                received_p_potentials = np.stack(self.__pending_p_potentials, axis=1)

                new_p_potential = np.sum(received_p_potentials, axis=1)

                if self.type != NeuronType.OUTPUT:
                    new_p_potential = np.where(np.greater(new_p_potential, 0), new_p_potential, 0)
                    # new_p_potential = np.where(np.greater(new_p_potential, 0), 1, 0)

        self.__p_potential = new_p_potential
        self.__pending_p_potentials = []
    
    def __update_e_potential(self):

        if self.type != NeuronType.LOSS:

            new_e_potential = np.zeros_like(self.e_potential)

            if len(self.__pending_e_potentials) > 0:

                received_e_potentials = np.stack(self.__pending_e_potentials, axis=1)

                new_e_potential = np.sum(received_e_potentials, axis=1)
            
            self.__e_potential = new_e_potential
            
        self.__pending_e_potentials = []
