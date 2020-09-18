import nltk
nltk.download('wordnet')
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('vader_lexicon')
nltk.download('punkt')
import spacy
import pandas as pd
import json
from analys import *
from text_analys import *
import CV_TF_Word2Vec_Anal
import LDA_topic_model
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.features.build_features import *
import re
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import *
import numpy as np
spacy.load('en')
from spacy.lang.en import English
parser = English()
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from gensim import corpora
import pickle
import pyLDAvis
import pyLDAvis.gensim
from modelanalys import *

def LDA_topic_anal(top5_drug_txt_parse,newpath_1):
    top5_drug_txt_lda= {}
    for (drug_name, total_txt) in top5_drug_txt_parse.items():
        text_data =[]
        for line in total_txt:
            tokens = prepare_text_for_lda(line)
            if random.random() > .99:
    #             print (line)
    #             print(tokens)
                text_data.append(tokens)
    #     print (text_data)
        top5_drug_txt_lda[drug_name]=text_data

    print (top5_drug_txt_lda['Zoloft'])

    # LDA with Gensim
    # First, we are creating a dictionary from the data, then convert to bag-of-words corpus and save the dictionary and corpus for future use.

    for (drug_name, text_data) in top5_drug_txt_lda.items():
        print (drug_name)
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(text) for text in text_data]
        # Dumped on total corpus & Dictionary
        pkn ='%s/corpus%s.pkl' %(newpath_1,drug_name)
        pickle.dump(corpus, open(pkn, 'wb'))
        dkn ='%s/dictionary%s.gensim' %(newpath_1,drug_name)
        dictionary.save(dkn)

        NUM_TOPICS = 5
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
        nm = '%s/model5%s.gensim' %(newpath_1,drug_name)
        ldamodel.save(nm)
        topics = ldamodel.print_topics(num_words=4)
        for topic in topics:
            print(topic)

    drug_name ='Zoloft'
    pkn ='%s/corpus%s.pkl' %(newpath_1,drug_name)
    dkn ='%s/dictionary%s.gensim' %(newpath_1,drug_name)
    dictionary = gensim.corpora.Dictionary.load(dkn)
    corpus = pickle.load(open(pkn, 'rb'))
    nm = '%s/model5%s.gensim' %(newpath_1,drug_name)
    lda = gensim.models.ldamodel.LdaModel.load(nm)
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=True)
    pyLDAvis.display(lda_display)

    # 80]:

    drug_name ='Prozac'
    pkn ='%s/corpus%s.pkl' %(newpath_1,drug_name)
    dkn ='%s/dictionary%s.gensim' %(newpath_1,drug_name)
    dictionary = gensim.corpora.Dictionary.load(dkn)
    corpus = pickle.load(open(pkn, 'rb'))
    nm = '%s/model5%s.gensim' %(newpath_1,drug_name)
    lda = gensim.models.ldamodel.LdaModel.load(nm)
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=True)
    pyLDAvis.display(lda_display)

    # 81]:
    drug_name ='Lexapro'
    pkn ='%s/corpus%s.pkl' %(newpath_1,drug_name)
    dkn ='%s/dictionary%s.gensim' %(newpath_1,drug_name)
    dictionary = gensim.corpora.Dictionary.load(dkn)
    corpus = pickle.load(open(pkn, 'rb'))
    nm = '%s/model5%s.gensim' %(newpath_1,drug_name)
    lda = gensim.models.ldamodel.LdaModel.load(nm)
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=True)
    pyLDAvis.display(lda_display)

    drug_name ='Effexor'
    pkn ='%s/corpus%s.pkl' %(newpath_1,drug_name)
    dkn ='%s/dictionary%s.gensim' %(newpath_1,drug_name)
    dictionary = gensim.corpora.Dictionary.load(dkn)
    corpus = pickle.load(open(pkn, 'rb'))
    nm = '%s/model5%s.gensim' %(newpath_1,drug_name)
    lda = gensim.models.ldamodel.LdaModel.load(nm)
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=True)
    pyLDAvis.display(lda_display)

    drug_name ='Wellbutrin'
    pkn ='%s/corpus%s.pkl' %(newpath_1,drug_name)
    dkn ='%s/dictionary%s.gensim' %(newpath_1,drug_name)
    dictionary = gensim.corpora.Dictionary.load(dkn)
    corpus = pickle.load(open(pkn, 'rb'))
    nm = '%s/model5%s.gensim' %(newpath_1, drug_name)
    lda = gensim.models.ldamodel.LdaModel.load(nm)
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=True)
    pyLDAvis.display(lda_display)

    #  ]ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
    ldamodel.save('%s/model3.gensim'%(newpath_1))
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
    ldamodel.save('%s/model10.gensim'%(newpath_1))
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
    # pyLDAvis
    # pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.
    # Visualizing 5 topics:

    dictionary = gensim.corpora.Dictionary.load('%s/dictionary.gensim'%(newpath_1))
    corpus = pickle.load(open('%s/corpus.pkl'%(newpath_1), 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('%s/model5.gensim'%(newpath_1))
    import pyLDAvis.gensim
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)
    # Saliency: a measure of how much the term tells you about the topic.
    # Relevance: a weighted average of the probability of the word given the topic and the word given the topic normalized by the probability of the topic.
    # The size of the bubble measures the importance of the topics, relative to the data.
    # First, we got the most salient terms, means terms mostly tell us about what’s going on relative to the topics. We can also look at individual topic.
    # Visualizing 3 topics:

    lda3 = gensim.models.ldamodel.LdaModel.load('%s/model3.gensim'%(newpath_1))
    lda_display3 = pyLDAvis.gensim.prepare(lda3, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display3)

    # Visualizing 10 topics:
    lda10 = gensim.models.ldamodel.LdaModel.load('%s/model10.gensim'%(newpath_1))
    lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display10)
