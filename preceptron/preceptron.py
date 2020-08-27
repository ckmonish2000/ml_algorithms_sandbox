import numpy as np

class Preceptron:
    def __init__(self,lr=0.001,epochs=10000):
        self.lr=lr
        self.epochs=epochs
        self.activation=self.unit_step_func
        self.weight=None
        self.bias=None

    def fit(self,X,y):
        pass

    def predict(self,X):
        pass

    def unit_step_func(self,x):
        return np.where(x>=0,1,0)