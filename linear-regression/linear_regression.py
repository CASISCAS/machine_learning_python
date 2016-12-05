import matplotlib.pyplot as plt
import numpy as np
from util.tools import scale

from sklearn import datasets, linear_model

"""
linear regression demo :
1. step by step from scratch LR support three fit methods. when use normal equation methods,remember to set iter to 0.
2. use sklearn LR for comparing.

"""
class LinearRegression(object):
    def __init__(self, alpha=0.001, n_iter=50, fit_alg='sgd'):
        self.fit_methods = {'sgd': self.fit_sgd, 'fit_batch': self.fit_batch, 'equation': self.equation_fit}
        self.alpha = alpha
        self.n_iter = n_iter
        self.fit_alg = self.fit_methods[fit_alg]

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w_ = np.ones(X.shape[1])

        if self.n_iter ==0:
            self.equation_fit(X, y)
        else:
            for _ in range(self.n_iter):
                X, y = self._shuffle(X,y)
                self.fit_alg(X, y)

    def equation_fit(self, X, y):
        X_T= np.transpose(X)
        self.w_ = np.linalg.inv(np.dot(X_T,X)).dot(X_T).dot(y)

    def fit_batch(self, X, y):
        output = X.dot(self.w_)
        errors = y - output
        self.w_ += self.alpha * X.T.dot(errors)
        # print(sum(errors**2) / 2.0)

    def fit_sgd(self, X, y):
        X, y = self._shuffle(X, y)
        for x, target in zip(X, y):
            output = x.dot(self.w_)
            errors = target - output
            self.w_ += self.alpha * x.T.dot(errors)

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w_)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)


def plot_predict(X_test, y_test, y_predict):

    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_predict, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def handy_model_demo():
    diabetes = datasets.load_diabetes()
    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_X = scale(diabetes_X)
    diabetes_y = scale(diabetes.target)

    diabetes_X_train = diabetes_X[:-20]
    diabetes_y_train = diabetes_y[:-20]

    regr = LinearRegression(n_iter=0, fit_alg='equation')
    regr.fit(diabetes_X_train, diabetes_y_train)

    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_test = diabetes_y[-20:]
    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    plot_predict(diabetes_X_test,diabetes_y_test, regr.predict(diabetes_X_test))


def sklearn_model_demo():


    diabetes = datasets.load_diabetes()


    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

    plot_predict(diabetes_X_test,diabetes_y_test, regr.predict(diabetes_X_test))

if __name__ == '__main__':
    handy_model_demo()
    sklearn_model_demo()








