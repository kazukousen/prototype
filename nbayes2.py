import numpy as np
from abc import ABCMeta, abstractmethod

class BaseBinaryNaiveBayes(object):
    """
    Abstract Class for Naive Bayes whose classes and features are binary.
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self.pY_ = None
        self.pXgY_ = None
    
    @abstractmethod
    def fit(self, X, y):
        """
        Abstract method for fitting model
        
        Attributes
        ----------
        `pY_` : array_like, shape=(n_classes), dtype=float
            pmf of a class
        `pXgY_` : array_like, shape(n_features, n_classes, n_fvalues), dtype=float
            pmf of feature values given a class
        """
        pass
    
    def predict(self, X):
        """
        Predict class
        
        Parameters
        ----------
        X: array_like, shape=(n_samples, n_features), dtype=int
            feature values of unseen samples
            
        Returns
        -----
        y: array_like, shape=(n_samples), dtype=int
            predict class labels
        """
        
        # constants
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # memory for return values
        y = np.empty(n_samples, dtype=np.int)
        
        # for each feature in X
        for i, xi in enumerate(X):
            
            # calc join probablity
            logpXY = np.log(self.pY_) + \
                np.sum(np.log(self.pXgY_[np.arange(n_features),xi,:]), axis=0)
            
            # predict class
            y[i] = np.argmax(logpXY)
        
        return y

class NaiveBayes1(BaseBinaryNaiveBayes):
    """
    Naive Bayes class (1)
    """
    
    def __init__(self):
        super(NaiveBayes1, self).__init__()
    
    def fit(self, X, y):
        """
        Fitting Model
        
        Parameters
        ----------
        X: array_like, shape=(n_samples, n_features), dtype=int
            feature values of training samples
        y: array_like, shape=(n_samples), dtype=int
            class labels of training samples
        """
        
        # constants
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = 2
        n_fvalues = 2
        
        # check the size of y
        if n_samples != len(y):
            raise ValueError('Mismatched number of samples.')
        
        # count up n[yi=y]
        nY = np.zeros(n_classes, dtype=np.int)
        for i in xrange(n_samples):
            nY[y[i]] += 1
        
        # calc pY_
        self.pY_ = np.empty(n_classes, dtype=np.float)
        for i in xrange(n_classes):
            self.pY_[i] = nY[i] / np.float(n_samples)
    
        # count up n[x_ij=xj, yi=y]
        nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=np.int)
        for i in xrange(n_samples):
            for j in xrange(n_features):
                nXY[j, X[i, j], y[i]] += 1
                
        # calc pXgY_
        self.pXgY_ = np.empty((n_features, n_fvalues, n_classes), dtype=np.float)
        for j in xrange(n_features):
            for xi in xrange(n_fvalues):
                for yi in xrange(n_classes):
                    self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / np.float(nY[yi])
    
class NaiveBayes2(BaseBinaryNaiveBayes):
    """
    Naive Bayes (2)
    """
    
    def __init__(self):
        super(NaiveBayes2, self).__init__()
        
    def fit(self, X, y):
        """
        Fitting Model
        
        Parameters
        ----------
        X: array_like, shape=(n_samples, n_features), dtype=int
            feature values of training samples
        y: array_like, shape=(n_samples), dtype=int
            class labels of training samples
        """
        
        # constants
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = 2
        n_fvalues = 2
        
        # check the size of y
        if n_samples != len(y):
            raise ValueError('Missmatched number of values.')
        
        # count up n[yi=y]
        nY = np.sum(y[:, np.newaxis] == np.arange(n_classes)[np.newaxis, :], axis=0)
        
        # calc pY_
        self.pY_ = np.true_divide(nY, n_samples)
        
        # count up n[x_ij=xj, yi=y]
        ary_xi = np.arange(n_fvalues)[np.newaxis, np.newaxis, :, np.newaxis]
        ary_yi = np.arange(n_classes)[np.newaxis, np.newaxis, np.newaxis, :]
        ary_y = y[:, np.newaxis, np.newaxis, np.newaxis]
        ary_X = X[:, :, np.newaxis, np.newaxis]

        nXY = np.sum(np.logical_and(ary_X == ary_xi, ary_y == ary_yi), axis=0)
        
        # calc pXgY_
        self.pXgY_ = np.true_divide(nXY, nY[np.newaxis, np.newaxis, :])
        
        