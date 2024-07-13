import numpy as np
from scipy.special import softmax
from .utils import C_EPSILON, loss, accuracy, shuffle, d_softmax, bias_scale, weight_scale
from .Neuron import Neuron, NeuronType
from .Synapse import Synapse
import random


class LSNPEP:

    # Must have at least the input and output layer
    MIN_LAYERS = 2

    def __init__(self,
                 layer_sizes: list,
                 learning_rate: float,
                 momentum: float = 0.9,
                 use_bias: bool = False):

        self.__layer_sizes = layer_sizes
        self.__n_layers = len(self.__layer_sizes)

        if self.__n_layers < LSNPEP.MIN_LAYERS:
            raise ValueError(f'Number of layers must be at least {LSNPEP.MIN_LAYERS}')

        self.__learning_rate = learning_rate

        self.__momentum = momentum

        self.__use_bias = use_bias

        self.__create_neurons()

        self.__create_synapses()
    
    def __create_neurons(self):

        self.__layers = [
            [Neuron(type=self.__get_layer_type(i),
                    id=(j if i == self.__n_layers-1 else None))
                               for j in range(self.__layer_sizes[i])]
                                                for i in range(self.__n_layers)]
        
        self.__loss_neuron = Neuron(type=NeuronType.LOSS,
                                    num_outputs=self.__layer_sizes[-1])
        
        self.__all_neurons = []
        for layer in self.__layers:
            for neuron in layer:
                self.__all_neurons.append(neuron)
        
        if self.__use_bias:
            self.__bias_neuron = Neuron(type=NeuronType.BIAS)
            self.__all_neurons.append(self.__bias_neuron)
        
        self.__all_neurons.append(self.__loss_neuron)
    
    def __create_synapses(self):

        self.__synapses = []

        # Regular synapses
        for i in range(self.__n_layers-1):
            l1_size = self.__layer_sizes[i]
            l2_size = self.__layer_sizes[i+1]
            for n1 in self.__layers[i]:
                for n2 in self.__layers[i+1]:
                    self.__synapses.append(
                        Synapse(pre=n1,
                                post=n2,
                                scale=weight_scale(l1_size, l2_size),
                                learning_rate=self.__learning_rate,
                                momentum=self.__momentum)
                    )
        
        if self.__use_bias:
            # Synapses with Bias Neuron
            for i in range(self.__n_layers-1):
                layer_size = self.__layer_sizes[i+1]
                for neuron in self.__layers[i+1]:
                    self.__synapses.append(
                        Synapse(pre=self.__bias_neuron,
                                post=neuron,
                                scale=bias_scale(layer_size),
                                learning_rate=self.__learning_rate,
                                momentum=self.__momentum)
                    )

        # Synapses with the Loss Neuron
        for n in self.__layers[-1]:
            self.__synapses.append(
                Synapse(pre=n,
                        post=self.__loss_neuron,
                        init_w=1.0,
                        learning_rate=self.__learning_rate)
            )

    def __get_layer_type(self, idx: int) -> NeuronType:
        
        if idx == 0:
            return NeuronType.INPUT
        elif idx == self.__n_layers-1:
            return NeuronType.OUTPUT
        else:
            return NeuronType.INNER

    def __reset(self, x: np.ndarray, y: np.ndarray = None):
        
        # Reset Input Layer with input x
        for i in range(x.shape[1]):
            self.__layers[0][i].reset(x[:, i].reshape((-1,)))

        # Reset Inner and Output Layers with zeros
        for layer in self.__layers[1:]:
            for neuron in layer:
                neuron.reset(np.zeros((x.shape[0],)))
        
        # Reset Bias Neuron
        if self.__use_bias:
            self.__bias_neuron.reset(np.zeros((x.shape[0],)))

        # Reset Loss Neuron
        if y is None:
            self.__loss_neuron.reset(np.zeros((x.shape[0],)))
        else:

            def d_loss_fn(p: np.ndarray) -> np.ndarray:

                softmax_of_p = softmax(p, axis=1)+C_EPSILON

                return np.squeeze(
                        np.matmul(
                            d_softmax(p),
                            np.expand_dims(y*(-1.0/softmax_of_p),
                                        axis=-1)))

            self.__loss_neuron.reset(
                        p=np.zeros((x.shape[0],)),
                        d_loss_fn=d_loss_fn)

        for syn in self.__synapses:
            syn.reset()

    def __predict_batch(self, x: np.ndarray) -> np.ndarray:

        self.__reset(x)

        for _ in range(self.__n_layers):

            for neuron in self.__all_neurons:
                neuron.execute()
            
            for synapse in self.__synapses:
                synapse.execute()
            
        # Collect p potentials when they reach the output layer
        y = np.stack([n.p_potential for n in self.__layers[-1]], axis=1)

        return y
    
    def __test_batch(self, x: np.ndarray,
                           y: np.ndarray) -> dict:

        self.__reset(x)

        for _ in range(self.__n_layers):

            for neuron in self.__all_neurons:
                neuron.execute()
            
            for synapse in self.__synapses:
                synapse.execute()
            
        # Collect p potentials when they reach the output layer
        outputs = np.stack([n.p_potential for n in self.__layers[-1]], axis=1)

        return {
            'loss': loss(y, outputs),
            'accuracy': accuracy(y, outputs)
        }
    
    def __train_batch(self, x: np.ndarray,
                            y: np.ndarray) -> dict:

        self.__reset(x, y)

        for i in range(2*self.__n_layers+1):

            for neuron in self.__all_neurons:
                neuron.execute()
            
            for synapse in self.__synapses:
                synapse.execute()
            
            # Collect p potentials when they reach the output layer
            if i == self.__n_layers-1:
                outputs = np.stack([n.p_potential for n in self.__layers[-1]], axis=1)

        return {
            'loss': loss(y, outputs),
            'accuracy': accuracy(y, outputs)
        }
    
    def predict(self, x: np.ndarray,
                      batch_size: int = 256) -> np.ndarray:
        
        num_samples = x.shape[0]
        num_features = x.shape[1]

        if num_features != self.__layer_sizes[0]:
            raise ValueError(f'Number of features do not match input neurons')

        y_pred = []

        start = 0
        while start < num_samples:

            end = min(num_samples, start + batch_size)

            xb = x[start:end, :]

            yb_pred = self.__predict_batch(xb)
            y_pred.append(yb_pred)

            start = end
        
        return np.concatenate(y_pred, axis=0)
    
    def test(self, x: np.ndarray,
                   y: np.ndarray,
                   batch_size: int = 256):
        
        num_samples = x.shape[0]
        num_features = x.shape[1]

        if num_features != self.__layer_sizes[0]:
            raise ValueError(f'Number of features do not match input neurons')
        
        loss_vals = []
        acc_vals = []

        start = 0
        while start < num_samples:

            end = min(num_samples, start + batch_size)

            xb = x[start:end, :]
            yb = y[start:end, :]

            vals = self.__test_batch(xb, yb)
            loss_vals.append(vals['loss'])
            acc_vals.append(vals['accuracy'])

            start = end

        print(f'   Loss: {np.mean(loss_vals)}')
        print(f'   Accuracy: {np.mean(acc_vals)}')
        print()
    
    def train(self, x: np.ndarray,
                    y: np.ndarray,
                    x_valid: np.ndarray,
                    y_valid: np.ndarray,
                    batch_size: np.int32 = 256,
                    epochs: np.int32 = 100):

        num_samples = x.shape[0]
        num_features = x.shape[1]

        if num_features != self.__layer_sizes[0]:
            raise ValueError(f'Number of features do not match input neurons')
        
        max_valid_acc = 0.0
        max_iter_no_improvement = 50
        stop_idx = 0

        for i in range(epochs):

            print(f'### Epoch {i+1} / {epochs}')

            x, y = shuffle(x, y)

            loss_vals = []
            acc_vals = []

            start = 0
            while start < num_samples:

                end = min(num_samples, start + batch_size)

                xb = x[start:end, :]
                yb = y[start:end, :]

                vals = self.__train_batch(xb, yb)
                loss_vals.append(vals['loss'])
                acc_vals.append(vals['accuracy'])

                start = end

            print(f'   Loss: {np.mean(loss_vals)}')
            print(f'   Accuracy: {np.mean(acc_vals)}')
            print()

            valid_acc = self.__test_batch(x_valid, y_valid)['accuracy']
            
            if valid_acc > max_valid_acc:

                max_valid_acc = valid_acc
                stop_idx = 0
            
            else:

                stop_idx += 1

                if stop_idx == max_iter_no_improvement:
                    print('-'*20)
                    print(f'Early stopping after {i+1} epochs')
                    print('-'*20)
                    break

    