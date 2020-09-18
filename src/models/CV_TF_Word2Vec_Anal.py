import nltk
nltk.download('wordnet')
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import *
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('vader_lexicon')
nltk.download('punkt')

import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()

import pandas as pd
import numpy as np
import re
import os
import glob
import operator

import gensim
from gensim.utils import simple_preprocess

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from lime import lime_text
from lime.lime_text import LimeTextExplainer
from nltk.tokenize import RegexpTokenizer

import sys
from pathlib import Path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.features.build_features import *
from modelanalys import *
from analys import *
from text_analys import *
import CV_TF_Word2Vec_Anal



def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    words = [word for word in words if word.lower() not in pills['BrandName'].values]
    #     words = [word for word in words if word.lower() not in pills['ChemName'].values]
    words = [word.lower() for word in words if word.isalpha()]
    words = [word.lower() for word in words if len(word) > 2]
    return words


def cv(data):
    count_vectorizer = CountVectorizer(analyzer=text_process)
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer


# ['Wellbutrin', 'Zoloft', 'Effexor', 'Lexapro', 'Prozac']
# ['Wellbutrin', 'Zoloft', 'Effexor', 'Lexapro', 'Prozac', 'Paxil', 'Cymbalta', 'Celexa', 'Remeron', 'Seroquel']
# using CV for vector transformation and feature extraction

def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(analyzer=text_process)
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer


def CV_anal(top_5_all_drugs_clean, newpath_1):
    # accuracy
    list_corpus = top_5_all_drugs_clean["body"].tolist()
    list_labels = top_5_all_drugs_clean["label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                        random_state=40)

    ####################################################################################Count vectorizer
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    plot = plot_LSA(X_train_counts, y_train)
    plot.savefig("%s/LSA_CV_post_process.png" % (newpath_1))
    plot.close()
    ########################################################################### Logistic regression/Accuracy/CM

    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)
    y_predicted_counts = clf.predict(X_test_counts)

    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    print(" = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    labels = ['Lexapro', 'Effexor', 'Wellbutrin', 'Prozac', 'Zoloft']
    cm = confusion_matrix(y_test, y_predicted_counts, labels)
    plot = plot_confusion_matrix(cm, normalize=True, title='Confusion matrix')
    plot.savefig("%s/cm_CV_LR.png" % (newpath_1))
    plot.close()
    importance = get_most_important_features(count_vectorizer, clf, 10)
    print(importance[1])
    for i in range(len(importance)):
        top_scores = [a[0] for a in importance[i]['tops']]
        top_words = [a[1] for a in importance[i]['tops']]
        bottom_scores = [a[0] for a in importance[i]['bottom']]
        bottom_words = [a[1] for a in importance[i]['bottom']]
        title = importance[i]['name']
        plot = plot_important_words(top_scores, top_words, bottom_scores, bottom_words,
                                    "Most important words for relevance", title)
        plot.savefig("%s/%simportance_CV_LR.png" % (newpath_1, title))
        plot.close()


##################################################################################### TF IDF transform
def TFIDF_anal(top_5_all_drugs_clean, newpath_1):
    # accuracy
    list_corpus = top_5_all_drugs_clean["body"].tolist()
    list_labels = top_5_all_drugs_clean["label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                        random_state=40)
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    plot = plot_LSA(X_train_tfidf, y_train)
    plot.savefig("%s/LSA_TFIDF_post_process.png" % (newpath_1))
    plot.close()
    ######################################################################################LR on TFIDF/CM,importance
    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                   multi_class='multinomial', n_jobs=-1, random_state=40)
    clf_tfidf.fit(X_train_tfidf, y_train)
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf,
                                                                           recall_tfidf, f1_tfidf))
    cm2 = confusion_matrix(y_test, y_predicted_tfidf)
    plot = plot_confusion_matrix(cm2, normalize=True, title='Confusion matrix', )
    plot.savefig("%s/cm_TFIDF_LR.png" % (newpath_1))
    plot.close()
    print("TFIDF confusion matrix")
    print(cm2)
    # These are just marginally better.. for obviuos reason: You don't see much of a separation
    importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)

    for i in range(len(importance_tfidf)):
        top_scores = [a[0] for a in importance_tfidf[i]['tops']]
        top_words = [a[1] for a in importance_tfidf[i]['tops']]
        bottom_scores = [a[0] for a in importance_tfidf[i]['bottom']]
        bottom_words = [a[1] for a in importance_tfidf[i]['bottom']]
        title = importance_tfidf[i]['name']
        plot = plot_important_words(top_scores, top_words, bottom_scores, bottom_words,
                                    "Most important words for relevance", title)
        plot.savefig("%s/%simportance_TFIDF_LR.png" % (newpath_1, title))
        plot.close()


#############################################################################################word2vectransform/CM/
# # Now on to Word2Vec representation: this incorporates the synonym structure
#
#
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


# top_5_all_drugs_clean = pd.DataFrame(columns = ['author', 'body', 'id', 'score', 'selftext_bysent', 'selftext_byWords', 'label'])

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['selftext_byWords'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                                          generate_missing=generate_missing))
    return list(embeddings)


def word2vec_pipeline(examples, word2vec):
    vector_store = word2vec
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_list = []
    for example in examples:
        example_tokens = tokenizer.tokenize(example)
        vectorized_example = get_average_word2vec(example_tokens, vector_store, generate_missing=False, k=300)
        tokenized_list.append(vectorized_example)
    return clf_w2v.predict_proba(tokenized_list)


def explain_one_instance(instance, class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(instance, word2vec_pipeline, word2vec, num_features=6)
    return exp


def visualize_one_exp(features, labels, index, class_names):
    exp = explain_one_instance(features[index], word2vec, class_names=class_names)
    print('Index: %d' % index)
    print('True class: %s' % [labels[index]])
    exp.show_in_notebook(text=True)


def get_statistical_explanation(test_set, sample_size, word2vec_pipeline, label_dict):
    sample_sentences = random.sample(test_set, sample_size)
    explainer = LimeTextExplainer()

    labels_to_sentences = defaultdict(list)
    contributors = defaultdict(dict)

    # First, find contributing words to each class
    for sentence in sample_sentences:
        probabilities = word2vec_pipeline([sentence])
        curr_label = probabilities[0].argmax()
        labels_to_sentences[curr_label].append(sentence)
        exp = explainer.explain_instance(sentence, word2vec_pipeline, num_features=6, labels=[curr_label])
        listed_explanation = exp.as_list(label=curr_label)

        for word, contributing_weight in listed_explanation:
            if word in contributors[curr_label]:
                contributors[curr_label][word].append(contributing_weight)
            else:
                contributors[curr_label][word] = [contributing_weight]

    # average each word's contribution to a class, and sort them by impact
    average_contributions = {}
    sorted_contributions = {}
    for label, lexica in contributors.items():
        curr_label = label
        curr_lexica = lexica
        average_contributions[curr_label] = pd.Series(index=curr_lexica.keys())
        for word, scores in curr_lexica.items():
            average_contributions[curr_label].loc[word] = np.sum(np.array(scores)) / sample_size
        detractors = average_contributions[curr_label].sort_values()
        supporters = average_contributions[curr_label].sort_values(ascending=False)
        sorted_contributions[label_dict[curr_label]] = {
            'detractors': detractors,
            'supporters': supporters
        }
    return sorted_contributions


def Word2Vec_anal(top_5_all_drugs_clean, newpath_1):
    # accuracy
    print("test_train_split")
    list_corpus = top_5_all_drugs_clean["body"].tolist()
    list_labels = top_5_all_drugs_clean["label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                        random_state=40)

    print("word2vecembedding")
    word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
    word2vec = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    embeddings = get_word2vec_embeddings(word2vec, top_5_all_drugs_clean)
    X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,
                                                                                            test_size=0.2,
                                                                                            random_state=40)
    plot = plot_LSA(embeddings, list_labels)
    plot.savefig("%s/Word2Vec_LSA_post_process" % (newpath_1))
    plot.close()

    plot_TSNE(embeddings, list_labels, newpath_1, "word2vec", savepath="TSNE_demo.csv", plot=True)
    plt.close()
    print("word2vecLR")
    clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                 multi_class='multinomial', random_state=40)
    clf_w2v.fit(X_train_word2vec, y_train_word2vec)
    y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

    accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec,
                                                                                      y_predicted_word2vec)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec,
                                                                           recall_word2vec, f1_word2vec))

    cm_w2v = confusion_matrix(y_test_word2vec, y_predicted_word2vec)
    plot = plot_confusion_matrix(cm_w2v, normalize=True, title='Confusion matrix')
    plot.savefig("%s/Word2Vec_cm_LR" % (newpath_1))
    plot.close()
    print("Word2Vec confusion matrix")
    print(cm_w2v)

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                                            random_state=40)
    vector_store = word2vec
    # X_train_counts, count_vectorizer = cv(X_train)
    # c = make_pipeline(count_vectorizer, clf)
    drug_class = {'Zoloft': 0, 'Lexapro': 1, 'Prozac': 2, 'Effexor': 3, 'Wellbutrin': 4}
    visualize_one_exp(X_test_data, y_test_data, 65, drug_class.keys())
    import random
    from collections import defaultdict
    random.seed(40)

    label_to_text = {0: 'Zoloft', 1: 'Lexapro', 2: 'Prozac', 3: 'Effexor', 4: 'Wellbutrin'}
    sorted_contributions = get_statistical_explanation(X_test_data, 100, word2vec_pipeline, label_to_text, word2vec)

    # First index is the class (Disaster)
    # Second index is 0 for detractors, 1 for supporters
    # Third is how many words we sample
    for i in (label_to_text.values()):
        title = i
        top_words = sorted_contributions[i]['supporters'][30:40].index.tolist()
        top_scores = sorted_contributions[i]['supporters'][30:40].tolist()
        bottom_words = sorted_contributions[i]['detractors'][30:40].index.tolist()
        bottom_scores = sorted_contributions[i]['detractors'][30:40].tolist()
        plot = plot_important_words(top_scores, top_words, bottom_scores, bottom_words,
                                    "Most important words for relevance", title)
        plot.savefig("%s/%simportance_Word2Vec_LR.png" % (newpath_1, title))
        plot.close()
