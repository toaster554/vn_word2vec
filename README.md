# vn_word2vec
Word2vec for Vietnamese using skip gram model  
**Packages used:** pandas, numpy, tensorflow 2.0, dill  
## **Usage:**
```
python word2vec.py [--embedding_dim] [--vocab_size] [--window_size] [--training_size] [--epochs] [--corpus_path]
```
**Argument Descriptions:**  
- **embedding_dim:** The dimensionality of your word embedding (Default: 100)  
- **vocab_size:** The size of the vocab (most common words) from your corpus (Default: 1000)  
- **window_size:** The window size that your skip gram model uses (Default: 1)  
- **training_size:** The number of sentences from corpus used for training (Default: 10000 which equals to around 700 mb of memory)  
- **epochs:** The number of epochs for training
- **corpus_path:** The path to your corpus (.pkl format) (Default: 'pickles\\corpus')
## **What does it do?**  
word2vec.py takes in a corpus of text (list of sentences) and performs word2vec embedding using skip grams on the most common words. The program returns two files called 'vecs.tsv' and 'meta.tsv' in embedding_visualization folder that can be fed into Tensorflow's [Embedding Projector](http://projector.tensorflow.org/). The program also saves the embedding weight in checkpoints.
