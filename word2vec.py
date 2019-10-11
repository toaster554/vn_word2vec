import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import vn_text_preprocess

embedding_dim = 100
vocab_size = 1000

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
