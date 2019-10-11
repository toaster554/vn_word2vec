import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import vn_text_preprocess as preprocess
import dill

embedding_dim = 100
vocab_size = 1000
# window size for skip gram
window_size = 1
corpus_path = 'pickles\\corpus.pkl'

# load corpus
corpus = []
with open(corpus_path, 'rb') as file:
	corpus = dill.load(file)

# preprocessing
data = preprocess.get_data(corpus, vocab_size, window_size)
dataset = tf.data.Dataset.from_tensor_slices((tf.one_hot(data[:,0], 1000),
	                                          tf.one_hot(data[:,1])))
dataset_batch = dataset.shuffle(1000).batch(128)

model = keras.Sequential([
	layers.Embedding(vocab_size, embedding_dim),
	layers.GlobalAveragePooling1D(),
	layers.Dense(1000, activation='sigmoid')
])

#model.summary()

model.compile(optimizer='adam',
	          loss='categorical_crossentropy',
	          metrics={'CategoricalAccuracy'})

history = model.fit(dataset_batch, epochs = num_epochs)
