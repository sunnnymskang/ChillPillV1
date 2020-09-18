import nltk
nltk.download('wordnet')
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('vader_lexicon')
nltk.download('punkt')
import pandas as pd
import numpy as np
import re
from analys import *
from text_analys import *

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.features.build_features import *
import re
import gensim

spacy.load('en')
from spacy.lang.en import English
parser = English()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
import numpy as np
from modelanalys import *
from sklearn.metrics import confusion_matrix

import string
from nltk.corpus import stopwords

docume = pd.read_excel("Pushift_doc.xlsx")
pills = pd.read_csv('antidepressants.txt')
pills['BrandName']= pills['Name'].str.split('\s+').str[0].str.strip()
pills['ChemName']= pills['Name'].str.split('\s+').str[1].str.strip("()").str.lower()

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
    words = [word.lower() for word in words if len(word)>2]
    return words


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
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
def word2vec_pipeline(examples,word2vec):
    vector_store= word2vec
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_list = []
    for example in examples:
        example_tokens = tokenizer.tokenize(example)
        vectorized_example = get_average_word2vec(example_tokens, vector_store, generate_missing=False, k=300)
        tokenized_list.append(vectorized_example)
    return clf_w2v.predict_proba(tokenized_list)


def LSVC_anal(top_5_all_drugs_clean,newpath_1):
    list_corpus = top_5_all_drugs_clean["body"].tolist()
    list_labels = top_5_all_drugs_clean["label"].tolist()

    #top_10_all_drugs_clean = pd.DataFrame(columns = ['author', 'body', 'id', 'score', 'selftext_bysent', 'selftext_byWords',
    #       'label', 'sentiment_body','rating_body'])
    top_5_all_drugs_clean['rating_body']= top_5_all_drugs_clean['rating_body'].astype(float)

    mean_score = top_5_all_drugs_clean.groupby(['label'], as_index=False)['rating_body'].mean()
    weight_dict = {k:v for (k,v) in mean_score.loc[:,['label','rating_body']].values}
    print (weight_dict)

    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, random_state=0, test_size=0.3)
    print (y_train[:3])

    # TF-IDF
    count_vect = CountVectorizer(analyzer= text_process)
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)

    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = tf_transformer.transform(X_test_counts)

    labels = LabelEncoder()
    y_train_labels_fit = labels.fit(y_train)
    y_train_lables_trf = labels.transform(y_train)

    # print(labels.classes_)
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    maps=list(labels.inverse_transform(range(len(labels.classes_))))
    values=range(len(labels.classes_))
    decodekeys= dict(zip(maps,values))
    print(decodekeys)
    weight_dict_encoded = {decodekeys[k]:v for (k,v) in weight_dict.items()}
    print (weight_dict_encoded)

    linear_svc = LinearSVC(class_weight = weight_dict_encoded)
    clf = linear_svc.fit(X_train_transformed,y_train_lables_trf)
    # linear_svc = LinearSVC()
    # clf = linear_svc.fit(X_train_transformed,y_train_lables_trf,top_5_all_drugs_clean['rating_body'].values)


    y_predicted_trf = clf.predict(X_test_transformed)
    y_test_labels_fit = labels.fit(y_test)
    # print ("y_test_labels_fit")
    # print (list(labels.classes_))
    y_test_lables_trf = labels.transform(y_test)
    # print ("y_test_lables_trf")
    # print (y_test_lables_trf[:10])
    # print ("y_test")
    # print (y_test[:10])
    # print ("y_test_lables_trf inverse")
    print (list(labels.inverse_transform(y_test_lables_trf[:10])))

    accuracy, precision, recall, f1 = get_metrics(y_test, labels.inverse_transform(y_predicted_trf))
    print(" = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    labels_1 =list(labels.classes_)
    cm = confusion_matrix(y_test, labels.inverse_transform(y_predicted_trf))
    plot = plot_confusion_matrix(cm, normalize=True,title='Confusion matrix',labels=labels_1)
    plot.savefig("%s/TFIDF_LSVC.png"%(newpath_1))
    plot.close()

    importance = get_most_important_features(count_vect , clf, 10)
    print (importance[1] )
    for i in range(len(importance)):
        top_scores = [a[0] for a in importance[i]['tops']]
        top_words = [a[1] for a in importance[i]['tops']]
        bottom_scores = [a[0] for a in importance[i]['bottom']]
        bottom_words = [a[1] for a in importance[i]['bottom']]
        title= importance[i]['name']
        plot= plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance", title)
        plot.savefig("%s/%simportance_TFIDF_SVC.png"%(newpath_1,title))
        plot.close()



    calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc,
                                            cv="prefit")

    calibrated_svc.fit(X_train_transformed,y_train_lables_trf)
    predicted = calibrated_svc.predict(X_test_transformed)

    to_predict = ["I have hyperinsomnia and social anxiety"]
    p_count = count_vect.transform(to_predict)
    p_tfidf = tf_transformer.transform(p_count)
    print('Average accuracy on test set={}'.format(np.mean(predicted == labels.transform(y_test))))
    print('Predicted probabilities of demo input string are')
    print(calibrated_svc.predict_proba(p_tfidf))

    result = pd.DataFrame(calibrated_svc.predict_proba(p_tfidf)*100, columns=labels.classes_)
    new = pd.melt(result,var_name='Drug name', value_name= 'likelihood(%)' )
    new= new.round(2)
    print(new)
    result_sorted =new.sort_values(by=['likelihood(%)'], ascending=False )
    result_sorted= result_sorted.reset_index(drop=True)
    result_sorted.rename(index= {0:'1st',1:'2nd',2:'3rd',3:'4th',4:'5th'}, inplace=True)
    result_sorted.index.name= "Rank"
    print (result_sorted)

    # # Save to file in the current working directory
    from sklearn.externals import joblib
    joblib.dump(calibrated_svc, '%s/linSVC_mode.joblib'%(newpath_1))
    joblib.dump(count_vect, '%s/CountVect_model.joblib'%(newpath_1))
    joblib.dump(tf_transformer, '%s/TFIDF_model.joblib'%(newpath_1))

    pkl_filename = "%s/labels.pkl" %(newpath_1)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(labels, file)
        file.close()




    # Word2vec
    word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
    word2vec = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    embeddings = get_word2vec_embeddings(word2vec, top_5_all_drugs_clean)
    X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels, test_size=0.2, random_state=40)

    # print(labels.classes_)
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    # maps=list(labels.inverse_transform(range(len(labels.classes_))))
    # values=range(len(labels.classes_))
    # decodekeys= dict(zip(maps,values))
    # print(decodekeys)
    # weight_dict_encoded = {decodekeys[k]:v for (k,v) in weight_dict.items()}
    # print (weight_dict_encoded)

    linear_svc = LinearSVC(class_weight = weight_dict)
    clf = linear_svc.fit(X_train_word2vec, y_train_word2vec)
    y_predicted_word2vec = clf.predict(X_test_word2vec)

    accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec, y_predicted_word2vec)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec,
                                                                           recall_word2vec, f1_word2vec))

    cm_w2v = confusion_matrix(y_test_word2vec, y_predicted_word2vec,labels= top_5_all_drugs_clean["label"].unique())
    plot = plot_confusion_matrix(cm_w2v, normalize=True, title='Confusion matrix')
    plot.savefig("%s/Word2Vec_SVC"%(newpath_1))
    plot.close()
    print("Word2Vec SVC confusion matrix")
    print(cm_w2v)


    calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc,
                                            cv="prefit")

    calibrated_svc.fit(X_train_word2vec, y_train_word2vec)
    predicted = calibrated_svc.predict(X_test_word2vec)

    to_predict = [["I", "have", "hyperinsomnia", "and", "social", "anxiety"]]
    p_word2vec = get_word2vec_embeddings(word2vec, to_predict)
    print('Predicted probabilities of demo input string are')
    print(calibrated_svc.predict_proba(p_tfidf))

    result = pd.DataFrame(calibrated_svc.predict_proba(p_tfidf)*100, columns=top_5_all_drugs_clean["label"].unique().values)
    new = pd.melt(result,var_name='Drug name', value_name= 'likelihood(%)' )
    new= new.round(2)
    print(new)
    result_sorted =new.sort_values(by=['likelihood(%)'], ascending=False )
    result_sorted= result_sorted.reset_index(drop=True)
    result_sorted.rename(index= {0:'1st',1:'2nd',2:'3rd',3:'4th',4:'5th'}, inplace=True)
    result_sorted.index.name= "Rank"
    print (result_sorted)

    # # Save to file in the current working directory
    from sklearn.externals import joblib
    joblib.dump(calibrated_svc, '%s/word2veclinSVC_model.joblib'%(newpath_1))

    pkl_filename = "%s/labels_word2vec_SVC.pkl" %(newpath_1)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(labels, file)
        file.close()
