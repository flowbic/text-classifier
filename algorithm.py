import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# Multinominal Naive Bayes
def multi_nb(X, y, test_data):  # X = data, y = labels
	clf = MultinomialNB()
	clf.fit(X, y)
	return clf.predict(test_data)


# Linear SVC (Support Vector Machines)
def lin_svc(X, y, test_data):
	clf = LinearSVC()
	clf.fit(X, y)
	return clf.predict(test_data)

