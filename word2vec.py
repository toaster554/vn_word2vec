import tensorflow as tf
import io
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import vn_text_preprocess as preprocess
import dill
import argparse

parse = argparse.ArgumentParser()
parser.add_argument("--embedding_dim", default = 100)
parser.add_argument("--vocab_size", default = 1000)
# window size for skip gram
parser.add_argument("--window_size", default = 1)
parser.add_argument("--corpus_path", default = 'pickles\\corpus.pkl')

args = parser.parse_args()

def main():
	embedding_dim = agrs.embedding_dim
	vocab_size = args.vocab_size
	window_size = args.window_size
	corpus_path = args.corpus_path
	# load corpus
    corpus = []
    with open(corpus_path, 'rb') as file:
    	corpus = dill.load(file)
    
    # preprocessing
    vocab, word2int, data = preprocess.get_data(corpus, vocab_size, window_size)
    dataset = tf.data.Dataset.from_tensor_slices((tf.one_hot(data[:,0], vocab_size),
    	                                          tf.one_hot(data[:,1], vocab_size)))
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

    # save embedding weights
    model.save_weights('checkpoints\\my_checkpoint')

    # embedding weights
    weights = model.layers[0].get_weights()[0]
    out_v = io.open('embedding_visualization\\vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('embedding_visualization\\meta.tsv', 'w', encoding='utf16')

    for idx, word in enumerate(word2int):
    	vec = weights[idx]
    	out_m.write(word + "\n")
    	out_v.write('\t'.join([str(x) for x in vec]) + '\n')

    out_m.close()
    out_v.close()

if __name__ == '__main__':
	main()