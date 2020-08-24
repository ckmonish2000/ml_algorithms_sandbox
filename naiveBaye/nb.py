import numpy as np


class NaiveBayes:
    def fit(self,X,y):
        n_samples,n_features=x.shape
        self._classes=np.unique(y)
        n_classes=len(self._classes)

        #init mean variance and priors
        self._mean=np.zeros((n_classes,n_features),dtype=np.float64)
        self._var=np.zeros((n_classes,n_features),dtype=np.float64)
        self.prior=np.zeros(n_classes,dtype=np.float64)


        for c in self._classes:
            X_c=X[c==y]
            self._mean[c,:]= X.mean(axis=0)
            self._var[c,:]= X.var(axis=0)
            self.prior[c]= X_c.shape[0]/ float(n_samples)




    def predict(self,X):
        predict=[self._predict(x) for x in X]
        return predict

    def _predict(self,x):
        posterios=[]
        
        for idx, c in enumerate(self._classes):
            prior=np.log(self.prior[idx])
            class_conditional=np.sum(np.log(self._pdf(idx,x)))
            posterio=prior+class_conditional
            posterios.append(posterio)
        return self._classes[np.argmax(posterios)]
            


    def _pdf(self,class_idx,x):
        mean=self._mean[class_idx]
        var=self._var[class_idx]
        numerator=np.exp(-(x-mean)**2/(2*var))
        denominator=np.sqrt(2*np.pi*var)
        return numerator/denominator
        
        
