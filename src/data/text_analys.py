# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy.integrate import odeint, solve_ivp
# import matplotlib as mpl
#
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import os
import nltk

# sns.set()  # over-write plt format with sns. plot made with plt will have sns formatting

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
# Lemmatization: get one type of verb
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

# Stop words - MAY HAVE TO EDIT THIS BETTER
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens
