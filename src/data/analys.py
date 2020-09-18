# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy.integrate import odeint, solve_ivp
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import os
import numpy as np
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import Englishs
parser = English()
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# nltk.download('vader_lexicon')
# nltk.download('punkt')
import operator
import gensim
import re
	# functionize
	###########################
	# Show pie chart of n prevalent words in the text



# FUNCTIONIZE
########åå###################################
# find the mathches of drug names from the dataframe['body']
# dict_a: dictionary name, store the result as dict_a[drugname]
# Pass : df['body']
# nm =drug_name å+ '_mentioned_all' + tag + source
def find_in_df(dict_a, df_subs, drug_name, filter_cols,nm,newpath_1,tag="save", col= 'body'):

        dict_a[drug_name] = pd.DataFrame(columns=filter_cols)
        dict_a[drug_name][filter_cols] = df_subs[filter_cols][df_subs[col].str.contains(drug_name, flags=re.IGNORECASE, regex=True)]
        print(len(dict_a[drug_name]))
        print("%f percent out of all comments " % (100 * len(dict_a[drug_name]) / len(df_subs)))
        print(dict_a[drug_name].describe())
        if tag =="save":
            if len(nm)==0:
                pth = newpath_1
            else:
                pth = newpath_1 + "/%s.pkl" % (nm)
            print ("pth:"+ pth)
            dict_a[drug_name].to_pickle(pth)
        else:
            pass
        return dict_a
##############################################


# FUNCTIONIZE: Top_n_ones
############################
def top_k_inDF(dict_a,k):
	# List of tuples (drug nane, dataframe for each drugs )
	top_k_drug_ment_sub = sorted(dict_a.items(), key=lambda x: len(x[1]),reverse=True)[0:k]
	top_k_drugs=  [x for (x,df) in top_k_drug_ment_sub]
	return top_k_drug_ment_sub, top_k_drugs
############################

#FUNCTIONIZE:Pickle the list of tuples
#######################
# Pickle information about top k mentioned drugs
# dict_a = data to pickle
# top_k_drug_ment_sub: list of tuple (drug_name, df); get name
# nm: file name
# new_path_1: path to file
def pickl_list_tupl( top_k_drug_ment_sub, nm, newpath_1):
	for (x, df) in top_k_drug_ment_sub:
		name = x + '_top5_' + nm
		df.to_pickle(newpath_1 + "/%s.pkl" % (name))
##########################

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




