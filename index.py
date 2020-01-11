import numpy as np
from utils import *
from algorithm import *

texts, labels = get_data('wikipedia_300/wikipedia_300.csv')
vectors = create_document_vectors(texts)
np_vectors = np.array(vectors)
print(np_vectors.shape)
multi_nb(np_vectors, labels)
lin_svc(np_vectors, labels)
