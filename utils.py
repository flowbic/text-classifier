import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Get data from file
def get_data(source):
	arr_list = []
	sentence_list = []
	data = np.genfromtxt(source, delimiter='",', skip_header=1, dtype=str, usecols=(0,1))
	for doc in data[:, 0]:
		sentence = doc[1: -1]
		arr = sentence.split(' ')  # Removes first comma and last extra char, then makes it to array
		sentence_list.append(sentence)  # appends as a sting
		arr_list.append(arr)  # appends as an array (each word = one element)

	return arr_list, sentence_list, data[:, 1]


def create_tf_idf_vectors(sentences):
	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(sentences)
	dense = vectors.todense()  # Result into matrix
	dense_list = dense.tolist()  # Matrix into list
	return dense_list


def create_document_vectors(data):
	vector_list = create_vector_list(data)  # The bag of words
	vectors = []
	for doc in data:
		vectors.append(create_binary_vector(vector_list, doc))  # Puts each document as vector, in list

	return vectors


# Creates one list with all words in all documents
def create_vector_list(data):
	s = set()
	for doc in data:
		for word in doc:
			s.add(word)

	return list(s)


# Makes binary vector from the bag of words and the array-document
def create_binary_vector(all_words_list, current_document):
	vector_list = []
	doc_set = set(current_document)
	for word in all_words_list:
		if word in doc_set:
			vector_list.append(1)
		else:
			vector_list.append(0)

	return vector_list


def accuracy_score(preds, y):
	return np.mean(preds == y)
