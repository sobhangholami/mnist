import numpy as np
from layers import *
import pandas as pd


class Model:
    
    
    def __init__(self, info: str):
        self.address = info
        df = pd.read_csv(self.address).T
        self.layers = [Flatten()]
        for i in df:
            AF = df[i].loc['activation_fn']
            pws = np.load(df[i].loc['perceptron_weights_path'])
            pbs = np.load(df[i].loc['perceptron_biases_path'])
            bnws = np.load(df[i].loc['bn_weights_path'])
            bnbs = np.load(df[i].loc['bn_biases_path'])
            bnm = np.load(df[i].loc['bn_mean_path'])
            bnv = np.load(df[i].loc['bn_var_path'])
            self.layers.append(Perceptron(pws,pbs))
            self.layers.append(BatchNorm1D(bnws,bnbs,bnm,bnv))
            if AF == 'relu':
                self.layers.append(Relu())
            else : self.layers.append(Softmax())


    def forward(self, x: np.ndarray, raw_output=False):
        input = self.process_input(x)
        for i in self.layers:
            output = i(input)
            input = output
        if raw_output==False:
            return output.argmax(axis=1)
        else: return output

    def process_input(self, x):
        if len(x.shape)>2:
            pass
        else:
            x = np.array([x])
        return x/255.0

    def __call__(self, x, raw_output=False):
        return self.forward(x,raw_output)

    def __repr__(self):
        return '\n'.join([str(i) for i in self.layers])