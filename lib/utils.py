import numpy as np
from random import shuffle as py_shuffle
from scipy.special import softmax


C_EPSILON = 1e-10

def bias_scale(layer_size: int) -> float:
    return weight_scale(1.0, layer_size)

def weight_scale(layer1_size: int, layer2_size) -> float:
    return np.sqrt(2.0/(layer1_size+layer2_size))

def ordered(t: tuple) -> tuple:

    i, j = t

    if i <= j:
        return (i, j)
    else:
        return (j, i)
    
def subset(lst: list, p: float = 0.9) -> list:

    py_shuffle(lst)
    n = int(len(lst) * p)

    return lst[:n]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def keep_in_range(x: float, low: float = 0.0, high: float = 1.0) -> float:
    if x < low:
        return low
    elif x > high:
        return high
    else:
        return x

def d_softmax(x: np.ndarray) -> np.ndarray:
    s = softmax(x, axis=1)
    D = np.stack([np.diag(s[i, :]) for i in range(s.shape[0])], axis=0)
    comb = np.matmul(np.expand_dims(s, 2), np.expand_dims(s, 1))
    return D - comb

def loss(true_labels: np.ndarray, outputs: np.ndarray) -> float:
    softmax_output = softmax(outputs, axis=1)
    a = softmax_output[true_labels.astype(np.bool_)] + C_EPSILON
    return np.mean(-np.log(a))

def accuracy(true_labels: list, predicted_labels: list) -> float:
    return (np.argmax(true_labels, axis=1) == np.argmax(predicted_labels, axis=1)).astype(np.float32).mean()

def scale_data(p: float, f_min: float, f_max: float, x_min: float, x_max: float) -> float:

    return p * (f_max-f_min) / (x_max-x_min) + (f_min*x_max - f_max*x_min) / (x_max - x_min)

def scale_dataset(x: np.ndarray, f_min: float, f_max: float) -> np.ndarray:

    x_min = np.min(x, axis=0).reshape((1, -1))
    x_max = np.max(x, axis=0).reshape((1, -1))

    return scale_data(x, f_min, f_max, x_min, x_max)

def labels_to_one_hot(x: np.ndarray) -> np.ndarray:
    n_classes = int(x.max()+1)
    return np.eye(n_classes)[x.astype(np.int32)]

def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape((-1, 28*28))

def polinomial_terms_3(input_potentials: list) -> list:
    
    new_input_potentials = []

    for input_potential in input_potentials:

        new_input_potential = []
        for p1 in input_potential:
            new_input_potential.append(p1)
            for p2 in input_potential:
                new_input_potential.append(p1*p2)
                for p3 in input_potential:
                    new_input_potential.append(p1*p2*p3)
        
        new_input_potentials.append(new_input_potential)
    
    return new_input_potentials

def shuffle(x: np.ndarray, y: np.ndarray) -> tuple:
    random_idx = np.arange(x.shape[0])
    np.random.shuffle(random_idx)
    return (x[random_idx], y[random_idx])

def chars_to_indices(y: np.ndarray) -> np.ndarray:

    y = np.array(y).reshape((-1,))

    char_id = 0
    char_map = {}

    for ch in y:
        if ch not in char_map:
            char_map[ch] = char_id
            char_id += 1
    
    for i in range(y.shape[0]):
        y[i] = char_map[y[i]]
    
    y = np.array(y, dtype=np.float32)
    return y
