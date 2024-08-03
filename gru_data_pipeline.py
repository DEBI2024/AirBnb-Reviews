"""This module preprocesses the dataset, download glove weights

Author: AbdElRhman ElMoghazy
Date 26-07-2024
"""
import os
import zipfile
import wget
import pandas as pd
import numpy as np
from sklearn.utils import resample

import tensorflow.data as tf_data # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


def get_glove():
    """download glove embedding pre-trained weights"""
    wget.download("https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip")

    with zipfile.ZipFile("glove.6B.zip", 'r') as zip_ref:
        zip_ref.extractall(".")


def get_input_paths():
    """get the paths to the input files , training and testing for each city"""

    paths = []
    for dirname, _, filenames in os.walk('./dataset'):
        for filename in filenames:
            if "csv" in filename:
                print(os.path.join(dirname, filename))
                paths.append(os.path.join(dirname, filename))

    return paths


def read_data(paths):
    """read the training and testing files into lists of dataframes"""

    test_dfs = []
    train_dfs = []
    for path in paths:
        if "test" in path:
            test_dfs.append(pd.read_csv(path, index_col=0))
        elif "train" in path:
            train_dfs.append(pd.read_csv(path, index_col=0))

    return train_dfs, test_dfs


def balance_data(data):
    """balance the data with refrence to the negative class"""

    negative_data = data.query("polarity_class == 0")
    positive_data = data.query("polarity_class == 1")

    positive_data = resample(positive_data,
                             replace=False,
                             n_samples=len(negative_data),
                             random_state=42)

    data_balanced = pd.concat([positive_data, negative_data])
    data_balanced = data_balanced.sample(frac=1)

    return data_balanced


def concat_data(train_dfs, test_dfs):
    """concatenate all training dataframes together and same for testing"""

    train_data = pd.concat(train_dfs)
    test_data = pd.concat(test_dfs)

    return train_data, test_data


def pad_data(max_len, tokenized_sequences):
    """pad sequences with post padding to match the max_length of the sequence
    Args:
      max_len: the maximum length of a sequence
      tokenized_sequences: the tokenized data to be padded
    Returns:
      padded_data: the sequences after padding
    """

    padded_data = pad_sequences(
        tokenized_sequences,
        maxlen=max_len,
        padding='post')

    return padded_data

def remove_stop_words(text_list):

  clean_comments = []
  for comment in text_list:
    clean_comment = remove_stopwords(comment)
    clean_comments.append(clean_comment)

  return clean_comments

def tokenize_data(text_lst, vocab_size=None, tokenizer=None, lemmatize = False):
    """tokenize the raw comments to get index sequences

    Args:
      text_lst: the list of raw comments
      vocab_size: the desired size of the vocab, default None when used to fit a new tokenizer
      tokenizer: the tokenizer used to tokenize the data, none if tokenizer is to be initialized

    Returns:
      tokenizer: tokenizer used with the data
      tokenized_sequences: the tokenized data
      """

    text_comments = text_lst.copy()

    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(text_lst)
    
    if lemmatize:
      lemmatizer = WordNetLemmatizer()
      lemmatized_sequences = []
      for sequence in text_lst:
        sequence_lemma = []
        for word in sequence:
          sequence_lemma.append(lemmatizer.lemmatize(word))
        lemmatized_sequences.append(" ".join(sequence_lemma))
      text_comments = lemmatized_sequences
    
    tokenized_sequences = tokenizer.texts_to_sequences(text_comments)

    return tokenizer, tokenized_sequences


def get_word_index(data):
    """get word indexes and vocabulary for usage with GLOVE"""

    vectorizer = keras.layers.TextVectorization(
        max_tokens=20000, output_sequence_length=100)

    text_ds = tf_data.Dataset.from_tensor_slices(data).batch(128)
    vectorizer.adapt(text_ds)

    vocabulary = vectorizer.get_vocabulary()
    word_index = dict(zip(vocabulary, range(len(vocabulary))))

    return word_index, vocabulary


def get_embeddings_indexes(path_to_glove):
    """get the word indexs from glove"""
    embeddings_index = {}
    with open(path_to_glove, encoding="utf-8") as file:
        for line in file:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    return embeddings_index


def initialize_glove(
        word_index,
        num_tokens,
        embedding_dim,
        path_to_glove):
    """initialize glove embedding matrix"""
    try:
        embeddings_index = get_embeddings_indexes(path_to_glove)
    except BaseException:
        print("didn't find imbeddings file")

    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def process_data(remove_stop, lemmatize):
    """apply all preprocessing steps
    Returns:
      train_data_raw: raw training dataframe before filtering
      test_data_raw: raw test dataframe
      x_train: training data after resampling and tokenization
      x_test: testing data after preprocessing
      y_train: y values of training data
      y_test: y_values of testing data
    """

    paths = get_input_paths()
    train_dfs, test_dfs = read_data(paths)

    # concatenating the data from all cities in the dataset
    train_data_raw, test_data_raw = concat_data(train_dfs, test_dfs)

    # balance the positive and negative classes
    balanced_train = balance_data(train_data_raw)

    y_train = balanced_train.polarity_class
    balanced_train = balanced_train.drop("polarity_class", axis=1)

    y_test = test_data_raw.polarity_class
    test_data_raw = test_data_raw.drop("polarity_class", axis=1)

    # converting data type of comments to str as some entries are just numbers
    balanced_train.comments = balanced_train.comments.astype(str)
    test_data_raw.comments = test_data_raw.comments.astype(str)

    if remove_stop:
    # remove stop words
      train_comments = remove_stop_words(balanced_train.comments)
      test_comments = remove_stop_words(test_data_raw.comments)
    else:
      train_comments = balanced_train.comments
      test_comments = test_data_raw.comments

    # tokenizing data
    tokenizer, tokenized_train = tokenize_data(train_comments,
                                               vocab_size=40000,
                                               tokenizer=None,
                                               lemmatize = lemmatize)

    _, tokenized_test = tokenize_data(
        test_comments, tokenizer=tokenizer, lemmatize = lemmatize)
  
    # padding sequences
    x_train = pad_data(max_len=100, tokenized_sequences=tokenized_train)
    x_test = pad_data(max_len=100, tokenized_sequences=tokenized_test)

    return train_data_raw, test_data_raw, x_train, x_test, y_train, y_test


def main(remove_stop = False, lemmatize = False):
    """apply all preprocessing steps"""

    path_to_glove = "./glove.6B.100d.txt"

    if not os.path.exists(path_to_glove):
        get_glove()

    embedding_dim = 100

    train_data_raw, _, x_train, x_test, y_train, y_test = process_data(remove_stop, lemmatize)

    word_index, voc = get_word_index(train_data_raw.comments.astype(str))
    num_tokens = len(voc) + 2
    embedding_matrix = initialize_glove(
        word_index,
        num_tokens,
        embedding_dim,
        path_to_glove)

    return x_train, x_test, y_train, y_test, embedding_matrix, num_tokens