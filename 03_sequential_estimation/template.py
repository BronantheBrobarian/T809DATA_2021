from numpy.random.mtrand import random
from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np
import random


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    
    X = [n, k]
    for i in range(n):
        for j in range(k):
            X[i, j] = np.random.multivariate_normal(mean, var, k)

    return X

def gen(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    T = [n, k]
    for i in range(k):
        T[n, i] = randint(-1, 1)
    print(T)


    X = [n, k]
    for i in range(k):
        X[n, i] = np.random.multivariate_normal(mean, X, var)
    
    return X


#mean = [0, 0]
#cov = [[1, 0], [0, 100]]  # diagonal covariance
#x, y = np.random.multivariate_normal(mean, cov, 5000).T
#plt.plot(x, y, 'x')
#plt.axis('equal')
#plt.show()

array = gen(2, 3, np.array([0, 1, -1]), 1.3)
print(f'array: {array}')

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    ...


def _plot_sequence_estimate():
    data = ...
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        ...
    plt.plot([e[0] for e in estimates], label='First dimension')
    ...
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    ...


def _plot_mean_square_error():
    ...


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    ...


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    ...
