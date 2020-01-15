import numpy as np
from algorithm import *
from utils import *


def crossval_predict(data, label_list, n):
	data_buckets, label_buckets = create_buckets(data, label_list, n) # Creates n buckets
	mnb_score = 0
	lsvc_score = 0
	for i, bucket in enumerate(data_buckets):
		print('Bag: ', i+1, ' of ', n)
		test_data = bucket
		test_labels = label_buckets[i]

		# Makes all buckets to one array, except bucket i
		train_labels = np.concatenate(np.delete(label_buckets, i, axis=0))  # For labels
		train_data = np.concatenate(np.delete(data_buckets, i, axis=0))  # For data

		# Training and predictions
		# returns scores
		mnb_predictions = multi_nb(train_data, train_labels, test_data)  # Multinomial Naive Bayes
		lsvc_predictions = lin_svc(train_data, train_labels, test_data)  # Linear SVC (Support Vector Machines)

		# Adds score for each prediction to variable, One variable for each algorithm
		mnb_score += accuracy_score(mnb_predictions, test_labels)
		lsvc_score += accuracy_score(lsvc_predictions, test_labels)

	return (mnb_score / n), (lsvc_score / n)  # returns each algorithms mean value


def create_buckets(data, label_list, n):
	data, label_list = shuffle(data, label_list)  # For getting randomized buckets

	# Divides the array to n arrays
	b_labels = np.array(np.array_split(label_list, n))
	b_data = np.array(np.array_split(data, n))

	return b_data, b_labels


def shuffle(data, label_list):
	indexes = np.arange(label_list.shape[0])  # creates an incrementing array from 0 to arr-length
	np.random.shuffle(indexes)  # Shuffles indexes. We have to do this so data and labels have the same random position
	return data[indexes], label_list[indexes]  # Returns data and labels in the new order

