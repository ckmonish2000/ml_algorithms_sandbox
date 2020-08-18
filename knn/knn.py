import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    dis=np.sqrt((np.sum(x1-x2)**2))


class KNN:
    def __init__(self,k=3):
        self.k=k

    def fit(self,X,y):
        self.X=X
        self.y=y
    
    def predict(self,x):
        predict_labels=[]
        return np.array(predict_labels)

    def _predict(self,x):
        distance=[euclidean_distance(x,x_train) for x_train in self.X]
        # sorting the elements with respect to 
        sort_distance=np.argsort(distance)[:self.k]
        # getting the labels for the nearest neighbours 
        nearest=[self.y[i] for i in sort_distance]
        # getting majority vote
        most_common=Counter(nearest).most_common(1)
        return most_common[0][0]

