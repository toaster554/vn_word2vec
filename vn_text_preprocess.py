import string
import os
import collections
import codecs
import pandas as pd
import numpy as np
# from vncorenlp import VnCoreNLP

# annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)
# annotator = VnCoreNLP("<path to vncorenlp-1.1.1.jar>", 
#                     annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')

# directory = 'C:\\Users\\Tenkyuu\\Desktop\\VNTC\\Data\\27Topics\\Ver1.1\\new train'

# given directory from root rootdir, extract sentences from .txt files
# return a list of sentences
# REQUIRES VnCoreNLP TO USE!!!
def get_text(rootdir):
    corpus = []
    # search for files within directory
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_path = os.path.join(subdir, file)
            corpus += extract_text_file(file_path)
    return corpus
                    
# given file path, extract sentences from .txt file and
# return a list of sentences
def extract_text_file(file_path):
    with codecs.open(file_path, 'r', encoding = 'utf16') as text_file:
        return annotator.tokenize(text_file.read())

def clean_corpus(corpus):
    punctuations = set(string.punctuation+'“”,.!:;/"@&)(\'\-–')
    # corpus[i] is a sentence
    for i in range(len(corpus)):
        # turn punctuations to empty strings
        corpus[i] = [s for s in corpus[i] if not any(c in punctuations for c in s)]
        # remove empty strings
        corpus[i] = [s for s in corpus[i] if s]
        # remove numbers
        corpus[i] = [x for x in corpus[i] if not any(c.isdigit() for c in x)]
        # remove words that start with upper letter
        corpus[i] = [x for x in corpus[i] if not x[0].isupper()]
    return corpus

# return Counter dict with words as keys
# and occurrences as values
def get_vocab(corpus):
    vocab = collections.Counter()
    for sentence in corpus:
        vocab += collections.Counter(dict(collections.Counter(sentence)))
        
    return vocab

# turn vocab to a dict with words as keys
# and id as values
def word_to_int(vocab):
    word2int = {}
    for i, word in enumerate(vocab):
        word2int[word] = i

    return word2int

# generate (input, label) data with skip gram model of window_size
def generate_data(corpus, window_size = 1):
    data = []
    for sentence in corpus:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(idx - window_size, 0) :
                            min(idx + window_size, len(sentence)) + 1]:
                if neighbor != word:
                    data.append([word, neighbor])
    
    return data

# preprocess corpus
def get_data(corpus, vocab_size, window_size):
    # corpus = get_text(directory)
    # remove strings with punctuations from corpus
    corpus = clean_corpus(corpus)
    # get list of words with their occurrences
    words = get_vocab(corpus)
    # get the (vocab_size) most common words
    vocab = set([i[0] for i in words.most_common()[:vocab_size]])
    word2int = word_to_int(vocab)
    # remove words not in vocab from corpus
    corpus = [[word for word in sentence if word in vocab] 
              for sentence in corpus]
    # process corpus to data using skip gram model
    data = generate_data(corpus, window_size)
    # turn data to pandas dataframe
    df = pd.DataFrame(data, columns = ['input', 'label'])
    # change words to their ids
    id_data = np.array([[word2int[row['input']], word2int[row['label']]] 
                       for index, row in df.iterrows()])
    # TODO: rewrite function with less operations
    return vocab, word2int, id_data

