import random

import numpy
from util.tools import scale
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegression(object):
    # initialize

    def __init__(self, X, Y, alpha=0.0005, lam=0.1, printIter=True):

        x = numpy.array(X)
        m, n = x.shape

        # normalize data
        x = scale(x)

        # add const column to X
        const = numpy.array([1] * m).reshape(m, 1)
        self.X = numpy.append(const, x, axis=1)

        self.Y = numpy.array(Y)
        self.alpha = alpha
        self.lam = lam
        self.theta = numpy.array([0.0] * (n + 1))

        self.printIter = printIter

    # transform function
    def __sigmoid(self, x):
        # m,n = x.shape
        # z = numpy.array([0.0]*(m*n)).reshape(m,n)
        z = 1.0 / (1.0 + numpy.exp((-1) * x))
        return z

    # caluclate cost
    def __costFunc(self):
        "calculate cost"
        m, n = self.X.shape
        h_theta = self.__sigmoid(numpy.dot(self.X, self.theta))

        cost1 = (-1) * self.Y * numpy.log(h_theta)
        cost2 = (1.0 - self.Y) * numpy.log(1.0 - h_theta)

        cost = (
                   sum(cost1 - cost2) + 0.5 * self.lam * sum(self.theta[1:] ** 2)) / m
        return cost

    # gradient descend
    def __gradientDescend(self, iters):
        """
        gradient descend:
        X: feature matrix
        Y: response
        theta: predict parameter
        alpha: learning rate
        lam: lambda, penality on theta
       """

        m, n = self.X.shape

        # print "m,n=" , m,n
        # print "theta", len(self.theta)

        for i in range(iters):
            theta_temp = self.theta

            # update theta[0]
            h_theta = self.__sigmoid(numpy.dot(self.X, self.theta))
            diff = h_theta - self.Y
            self.theta[0] = theta_temp[0] - self.alpha * \
                                            (1.0 / m) * sum(diff * self.X[:, 0])

            for j in range(1, n):
                val = theta_temp[
                          j] - self.alpha * (1.0 / m) * (sum(diff * self.X[:, j]) + self.lam * m * theta_temp[j])
                # print val
                self.theta[j] = val
            # calculate cost and print
            cost = self.__costFunc()

            if self.printIter:
                print("Iteration", i, "\tcost=", cost)
                # print "theta", self.theta

    # simple name
    def train(self, iters, printIter=True):
        self.printIter = printIter
        self.__gradientDescend(iters)

    # prediction
    def predict(self, X):

        # add const column
        m, n = X.shape
        x = numpy.array(X)
        x = (x - self.xMean) / self.xStd
        const = numpy.array([1] * m).reshape(m, 1)
        X = numpy.append(const, x, axis=1)

        pred = self.__sigmoid(numpy.dot(X, self.theta))
        numpy.putmask(pred, pred >= 0.5, 1.0)
        numpy.putmask(pred, pred < 0.5, 0.0)

        return pred


def load_data(filename='input.csv'):
    data = numpy.genfromtxt(filename, delimiter=',')
    # response is in the first column
    y = data[:, 0]
    x = data[:, 1:]

    # shuffle data
    m = len(y)
    index = numpy.arange(m)
    random.shuffle(index)
    x = x[index, :]
    y = y[index]

    return x, y


def demo():
    # n-fold cross validation

    X, Y = load_data()

    m = len(Y)
    nfold = 10
    foldSize = int(m / nfold)

    # arrage to store training and testing error
    trainErr = [0.0] * nfold
    testErr = [0.0] * nfold
    allIndex = range(0, m)
    for i in range(0, nfold):
        testIndex = range((foldSize * i), foldSize * (i + 1))
        trainIndex = list(set(allIndex) - set(testIndex))

        trainX = X[trainIndex, :]
        trainY = Y[trainIndex]
        testX = X[testIndex, :]
        testY = Y[testIndex]

        # set parameter
        alpha = 0.05
        lam = 0.1
        model = LogisticRegression(trainX, trainY, alpha, lam)
        model.train(400, printIter=False)

        trainPred = model.predict(trainX)
        trainErr[i] = float(sum(trainPred != trainY)) / len(trainY)

        testPred = model.predict(testX)
        testErr[i] = float(sum(testPred != testY)) / len(testY)

        print("train Err=", trainErr[i], "test Err=", testErr[i])

    print("summary:")
    print("average train err =", numpy.mean(trainErr) * 100, "%")
    print("average test err =", numpy.mean(testErr) * 100, "%")


def sklearn_demo():

    X, Y = load_data()

    kf = KFold(n_splits=10)

    model = LR(max_iter=400)
    train_errors = []
    scores = []

    for k, (train, test) in enumerate(kf.split(X, Y)):

        model.fit(X[train], Y[train])
        train_error = numpy.mean((model.predict(X[test])-Y[test])**2)
        score = model.score(X[test], Y[test])

        train_errors.append(train_error)
        scores.append(score)
        print("train error :{0:.5f}".format(train_error))
        print("[fold {0}] , score: {1:.5f}".format(k, score))

    print("summary:")
    print("average train err =", numpy.mean(train_errors) * 100, "%")
    print("average test score =", numpy.mean(scores) * 100, "%")


if __name__ == '__main__':
    # demo()
    sklearn_demo()