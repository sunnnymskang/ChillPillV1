import nltk
nltk.download('wordnet')
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('vader_lexicon')
nltk.download('punkt')
import spacy
import pandas as pd
from analys import *
import re
import matplotlib.pyplot as plt
from nltk.stem.porter import *

spacy.load('en')
from spacy.lang.en import English
parser = English()
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import itertools


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
#         print (class_index)
#         print (model.classes_[class_index])
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom,
            'name': model.classes_[class_index] }

    return classes


def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name, title):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]

    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(10, 10))

    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title(title, fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    return plt

def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter,labels=['']):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)


    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    return plt


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
#     print(color_column)
#     print(test_data.shape)
    print (lsa_scores[:,1])
    print(color_mapper)
    colors = ['red','green','yellow','blue','black','blue']

    # LSA scatter plot
    if plot:
        fig1 =plt.figure(figsize=(11, 11))
        fig1.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=color_column, cmap=matplotlib.colors.ListedColormap(colors))
#     red_patch = mpatches.Patch(color='orange', label='Irrelevant')
#     green_patch = mpatches.Patch(color='blue', label='Disaster')
    patch_1 = mpatches.Patch(color='red', label='Effexor')
    patch_2  = mpatches.Patch(color='green', label='Lexapro')
    patch_3  = mpatches.Patch(color='yellow', label='Wellbutrin')
    patch_4  = mpatches.Patch(color='blue', label='Prozac')
    patch_5  = mpatches.Patch(color='black', label='Zoloft')

    fig1.legend(handles=[patch_1, patch_2,patch_3,patch_4,patch_5], prop={'size': 20})

    return fig1

def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
#     print(color_column)
#     print(test_data.shape)
    print (lsa_scores[:,1])
    print(color_mapper)
    colors = ['red','green','yellow','blue','black','blue']

    # LSA scatter plot
    if plot:
        fig1 =plt.figure(figsize=(11, 11))
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=color_column, cmap=matplotlib.colors.ListedColormap(colors))
#     red_patch = mpatches.Patch(color='orange', label='Irrelevant')
#     green_patch = mpatches.Patch(color='blue', label='Disaster')
    patch_1 = mpatches.Patch(color='red', label='Effexor')
    patch_2  = mpatches.Patch(color='green', label='Lexapro')
    patch_3  = mpatches.Patch(color='yellow', label='Wellbutrin')
    patch_4  = mpatches.Patch(color='blue', label='Prozac')
    patch_5  = mpatches.Patch(color='black', label='Zoloft')

    plt.legend(handles=[patch_1, patch_2,patch_3,patch_4,patch_5], prop={'size': 20})

    return plt




# # Apply clustering instead of class names.
# from sklearn.cluster import KMeans
#
# clusters = KMeans(n_clusters=5)
# clusters.fit(docs)
#
# tsne = TSNEVisualizer()
# tsne.fit(docs, ["c{}".format(c) for c in clusters.labels_])
# tsne.poof()

def plot_TSNE(test_data, test_labels, newpath_1,kind, savepath="TSNE_demo.csv", plot=True):
    from sklearn.manifold import TSNE
    from yellowbrick.text import TSNEVisualizer
    tsne = TSNEVisualizer()
    tsne.fit(test_data, ["c{}".format(c) for c in test_labels])
    fnm = "%s/%stSNE.png" %(newpath_1,kind)
    tsne.poof()

    # return fig1
