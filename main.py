#!/usr/bin/env python3
"""Initial training of fake review model.

https://practicaldatascience.co.uk/machine-learning/how-to-build-a-fake-review-detection-model
"""
import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, \
                            precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('max_colwidth', None)
df = pd.read_csv('data/fake reviews dataset.csv', names=['category', 'rating', 'label', 'text'])

df['text'] = df['text'].str.replace('\n', ' ')
df['target'] = np.where(df['label']=='CG', 1, 0)

def punctuation_to_features(df, column):
    """Identify punctuation within a column and convert to a text representation.

    Args:
        df (object): Pandas dataframe.
        column (string): Name of column containing text.

    Returns:
        df[column]: Original column with punctuation converted to text,
                    i.e. "Wow!" > "Wow exclamation"

"""
    df[column] = df[column].replace('!', ' exclamation ')
    df[column] = df[column].replace('?', ' question ')
    df[column] = df[column].replace('\'', ' quotation ')
    df[column] = df[column].replace('\"', ' quotation ')

    return df[column]

df['text'] = punctuation_to_features(df, 'text')

nltk.download('punkt')

def tokenize(column):
    """Tokenizes a `pandas` dataframe column and returns a list of tokens.

    Args:
        column: `pandas` dataframe column (i.e. `df['text']`).

    Returns:
        tokens (list): Tokenized list, i.e. [Fourthbrain, MLO4, Capstone]

    """

    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]

df['tokenized'] = df.apply(lambda x: tokenize(x['text']), axis=1)

nltk.download('stopwords')

def remove_stopwords(tokenized_column):
    """Return a list of tokens with English stopwords removed.

    Args:
        column: Pandas dataframe column of tokenized data from `tokenize()`

    Returns:
        tokens (list): Tokenized list with stopwords removed.

    """
    stops = set(stopwords.words("english"))
    return [word for word in tokenized_column if not word in stops]

df['stopwords_removed'] = df.apply(lambda x: remove_stopwords(x['tokenized']), axis=1)

def apply_stemming(tokenized_column):
    """Return a list of tokens with Porter stemming applied.

    Args:
        column: Pandas dataframe column of tokenized data with stopwords removed.

    Returns:
        tokens (list): Tokenized list with words Porter stemmed.

    """

    stemmer = PorterStemmer()
    return [stemmer.stem(word).lower() for word in tokenized_column]

df['porter_stemmed'] = df.apply(lambda x: apply_stemming(x['stopwords_removed']), axis=1)

def rejoin_words(tokenized_column):
    return ( " ".join(tokenized_column))

df['all_text'] = df.apply(lambda x: rejoin_words(x['porter_stemmed']), axis=1)