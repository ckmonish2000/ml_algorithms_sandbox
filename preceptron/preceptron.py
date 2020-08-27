import numpy as np

class Preceptron:
    def __init__(self,lr=0.001,epochs=10000):
        self.lr=lr
        self.epochs=epochs
        self.activation=self.unit_step_func
        self.weight=None
        self.bias=None

    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weight=np.zeros(n_features)
        self.bias=0

        y_=[1 if i>0 else 0 for i in y]
        for epoch in range(self.epochs):
            for idx,x_i in enumerate(X):
                pred=np.dot(x_i,self.weight)+self.bias
                pred=self.activation(pred)
                update=self.lr*(y[idx]-pred)
                self.weight=update*x_i
                self.bias+= update



    def predict(self,X):
        linear=np.dot(X,self.weight )+self.bias
        pred=self.activation(linear)
        return pred

    def unit_step_func(self,x):
        return np.where(x>=0,1,0)