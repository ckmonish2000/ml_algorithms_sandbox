import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from knn import KNN

iris=datasets.load_iris()
features,targets=iris.data,iris.target

X_train,X_test,y_train,y_test=train_test_split(features,targets,test_size=0.2,random_state=1234)

model=KNN()
model.fit(X_train,y_train)
pred=model.predict(X_test)

acc=np.sum(pred==y_test)/len(y_test)
print(acc)

