from preceptron import Preceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
model=Preceptron()

def accuracy(y_true,y_pred):
    acc=np.sum(y_true==y_pred)/len(y_true)
    return acc


X,y=datasets.make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.05,random_state=2)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print(accuracy(y_test,pred))

