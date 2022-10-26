import re
from turtle import forward
from matplotlib.pyplot import axis
import numpy as np
import matplotlib.pyplot as plt

class Flatten:
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_data = np.array([i.flatten() for i in x])
        return x_data

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return  "Flatten"


class Perceptron:
    def __init__(self, ws: np.ndarray, bs: np.ndarray):
        self.ws = ws
        self.bs = bs

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x,self.ws.T)+self.bs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self):
        return 'Perceptron ({}, {})'.format(self.ws.shape[1],self.ws.shape[0])


class Softmax:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        l = np.array([np.exp(i - np.max(i))/(np.exp(i - np.max(i)).sum(axis=0)) for i in x])
        return l
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self):
        return  'Softmax Activation function'


class Relu:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        l = np.array([np.maximum(0,i) for i in x])
        return l

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self):
        return 'Relu Activation function'


class BatchNorm1D:
    def __init__(self, ws: np.ndarray, bs: np.ndarray, mean: np.ndarray, var: np.ndarray):
        self.ws = ws
        self.bs = bs
        self.mean = mean
        self.var = var
        self.n = ws.size
    def forward(self, x: np.ndarray):
        # l = np.array([(x[i,:]-self.mean[i])/(np.sqrt(self.var[i])) for i in range(len(self.mean))])
        # # return np.array([l[i,:]*self.ws[i]+self.bs[i] for i in range(len(self.ws))])
        # d = np.einsum("ij,i->ij", l, self.ws)
        # return d + self.bs.reshape(-1,1)
        norm = (x - self.mean) / np.sqrt(self.var)
        return np.einsum("bi, i->bi", norm, self.ws) + self.bs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self):
        return 'BatchNorm1D {}'.format(self.n)