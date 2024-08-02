'''
This module is used to preprocess the data before running topic modeling.

Author: Shahd Seddik
Date: 22-07-2024
'''

import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from tqdm import tqdm
import os
tqdm.pandas()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

DATASET_DIR = os.environ.get('DATASET_DIR', 'data')

def plot_sentiment_dist(df, city):
    plt.figure(figsize=(6, 4))
    # pie chart
    df['polarity_class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Sentiment Distribution in {city}')
    plt.show()

def detect_lang(text):
    '''Detect the language of a text using the langdetect library'''
    try:
        return detect(text)
    except:
        return 'unknown'

def plot_lang_dist(df, city):
    '''Plot the distribution of languages in the dataset'''
    plt.figure(figsize=(10, 4))
    sns.countplot(x='lang', data=df)
    plt.title(f'Language Distribution in {city}')
    plt.show()

# Stop words for English
stop_words = set(stopwords.words('english'))

def preprocess(text):
    '''Preprocess a text by removing HTML tags, punctuation, stop words, and lemmatizing the words'''
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

