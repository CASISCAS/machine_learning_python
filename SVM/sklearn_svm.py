
from sklearn import datasets
from sklearn import svm


iris = datasets.load_iris()
clf = svm.SVC()
X, y = iris.data, iris.target

clf.fit(X, y)

print(clf.predict(iris.data[:3]))