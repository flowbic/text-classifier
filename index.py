import numpy as np
from utils import *
from algorithm import *
from cross_validation import *

arrs, sentences, labels = get_data('wikipedia_300/wikipedia_300.csv') # Gets data from file
binary_vectors = np.array(create_document_vectors(arrs)) # binary vector lists
tf_idf_vectors = np.array(create_tf_idf_vectors(sentences))

print('Running binary approach...')
nmb_score, lsvc_score = crossval_predict(binary_vectors, labels, 10)
print('MultinomailNB score: ', nmb_score)
print('LinearSVC score: ', lsvc_score)
print('-------------------------------')

print('Running TF-IDF approach...')
nmb_score, lsvc_score = crossval_predict(tf_idf_vectors, labels, 10)
print('MultinomailNB TF-IDF score: ', nmb_score)
print('LinearSVC TF-IDF score: ', lsvc_score)
