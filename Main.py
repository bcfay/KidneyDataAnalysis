
import numpy as np
import arff
data = arff.load(open('chronic_kidney_disease_full.arff', 'rb'))
print(data)
data, meta = arff.loadarff('chronic_kidney_disease_full.arff')
print(data)
print(meta)


class LogReg:

    def __init__(self, loops, a=0.01):
        self.lr = a
        self.loops = loops



    def cost(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def sigmoid(z):
        y = 1 / (1 + np.exp(-z))
        return y

    def fit(self, X, y):

        # weights vector
        self.theta = np.zeros(X.shape[1])
        # weights training
        for i in range(self.loops):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

