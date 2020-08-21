import numpy as np

class LinearRegression:
    def __init__(self,lr=0.001,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        num_samples,num_features=X.shape
        self.weights=np.zeros(num_features)
        self.bias=0

        for _ in range(self.epochs):
            y_pred=np.dot(X,self.weights)+self.bias
            dw=(1/num_samples)* np.dot(X.T,(y_pred-y))
            db=(1/num_samples)* np.sum((y_pred-y))
            self.weights-=self.lr*dw
            self.bias-=self.lr*db
            
    def predict(self,X):
        y_pred=np.dot(X,self.weights)+self.bias
        return y_pred

