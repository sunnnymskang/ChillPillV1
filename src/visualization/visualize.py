# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
# nltk.download('wordnet')
import spacy

spacy.load('en')
from spacy.lang.en import English

parser = English()
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def plot_n_prev_rare(list,drug_name,n, newpath_1,opt= None):
    if opt == 'largest':
        plt.figure()
        print(pd.Series(list).describe())
        pd.Series(list).value_counts().nlargest(n).plot.pie(autopct='%.2f', fontsize=10, figsize=(6, 6))
        tit = drug_name + "20 prevalent words"
        plt.title(tit)
        plt.savefig("%s/%s.png"%(newpath_1,tit))
        plt.close()
    elif opt == 'smallest':
        plt.figure()
        pd.Series(list).value_counts().nsmallest(n).plot.pie(autopct='%.2f', fontsize=10, figsize=(6, 6))
        tit = drug_name + "20 prevalent words"
        plt.title(drug_name + "20 rare words")
        plt.savefig("%s/%s.png"%(newpath_1,tit))
        plt.close()
    else:
        print ("use either largest or smallest")
        print("Distribution of top 30 words as default ")
        plt.figure()
        print(pd.Series(list).describe())
        pd.Series(list).value_counts().nlargest(30).plot.pie(autopct='%.2f', fontsize=10, figsize=(6, 6))
        tit = drug_name + "20 prevalent words"
        plt.title(tit)
        plt.savefig("%s/%s.png"%(newpath_1,tit))
        plt.close()
