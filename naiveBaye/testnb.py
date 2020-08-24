from nb import NaiveBayes
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

def accuracy(pred,label):
    acc=np.sum(pred==label)/len(label)
    return acc

X,y=datasets.make_classification(n_samples=1000,n_features=4,n_classes=2,random_state=123)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)


model=NaiveBayes()
model.fit(X_train,y_train)
predict=model.predict(X_test)

acc=accuracy(predict,y_test)
print(acc)