import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def multi_nb(X, y): # X = data, y = labels
	clf = MultinomialNB()
	clf.fit(X, y)
	print(clf.predict(X[2:3]))


def lin_svc(X, y):
	clf = LinearSVC()
	clf.fit(X, y)
	print(clf.predict(X[2:3]))
