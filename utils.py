import numpy as np


def get_data(source):
	list = []
	data = np.genfromtxt(source, delimiter='",', skip_header=1, dtype=str, usecols=(0,1))
	for doc in data[:, 0]:
		arr = doc[1: -1].split(' ')  # Removes first comma and last extra char, then makes it to array
		list.append(arr)

	return list, data[:, 1]


def create_document_vectors(data):
	vector_list = create_vector_list(data)
	vectors = []
	for doc in data:
		vectors.append(create_binary_vector(vector_list, doc)) # Puts each document as vector, in list

	return vectors

# Creates one list with all words in all documents
def create_vector_list(data):
	s = set()
	for doc in data:
		for word in doc:
			s.add(word)

	return list(s)


def create_binary_vector(all_words_list, current_document):
	vector_list = []
	doc_set = set(current_document)
	for word in all_words_list:
		if word in doc_set:
			vector_list.append(1)
		else:
			vector_list.append(0)

	return vector_list

