import nltk

nltk.download('wordnet')
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('vader_lexicon')
nltk.download('punkt')
import pandas as pd
from analys import *
import matplotlib.pyplot as plt
import gensim

spacy.load('en')
from spacy.lang.en import English

parser = English()
from modelanalys import *
import sys
from pathlib import Path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.visualization.visualize import *


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def sent_to_score(sent):
    if sent == 'neu':
        score = 3
    elif sent == 'pos':
        score = 5
    else:
        # sent == 'neg':
        score = 1
    return score


def sentiment_analysis(message_text):
    # next, we initialize VADER so we can use it within our Python script
    sid = SentimentIntensityAnalyzer()

    # the variable 'message_text' now contains the text we will analyze.
    message_text = ''' %s''' % (message_text)

    # Calling the polarity_scores method on sid and passing in the message_text outputs a dictionary with negative, neutral, positive, and compound scores for the input text
    scores = sid.polarity_scores(message_text)
    scores.pop('compound')
    sentiment = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    sent, score = (tuple(sentiment)[0])
    rating_score = sent_to_score(sent)
    return sent, rating_score


def body_to_sent_words(df, clm):
    # Take the body of a dataframe and parse it to sentence and words
    # Store them in separate columns

    df['selftext_bysent'] = [[]] * len(df)
    df['selftext_byWords'] = [[]] * len(df)
    df['sentiment_body'] = [[]] * len(df)
    df['rating_body'] = [[]] * len(df)
    total_txt_data = []  # Sentence list
    total_txt_word = []  # words list

    for i in df[clm].index:
        # Parse into sentences
        parsed = tokenizer.tokenize(df.at[i, clm])
        df.at[i, 'selftext_bysent'] = parsed

        # Parse into words - create list of lists
        words = list(sent_to_words(df.at[i, 'selftext_bysent']))
        words_flatten = [item for sublist in words for item in sublist]

        df.at[i, 'selftext_byWords'] = words_flatten
        total_txt_word.extend(words_flatten)
        total_txt_data.extend(tokenizer.tokenize(df.loc[i]['body'])[:])

        sent, score = sentiment_analysis(df.at[i, 'body'])
        df.at[i, 'sentiment_body'] = sent
        df.at[i, 'rating_body'] = score
    return df, total_txt_data, total_txt_word


def sentanal(top_5_drug_ment_sub, newpath_1):
    """
        args:
        top_5_drug_ment_sub - list of tuple (drug_name, df)
        df: columns = []
        Take df
        Broke msg into sentences , words and score of sentiment on the whole message
        df is mutated
        returns parsed text data as a dictionary
    """
    text_data = []
    top5_drug_txt_parse = {}

    for drug_name, df in top_5_drug_ment_sub:
        print('\n')
        print(drug_name)
        df['label'] = drug_name
        # Create new column of selt text parsed into sentences
        df, total_txt_data, total_txt_word = body_to_sent_words(df, 'body')
        top5_drug_txt_parse[drug_name] = total_txt_data

        plot_n_prev_rare(total_txt_word, drug_name, 20, newpath_1, opt="largest")
        plot_n_prev_rare(total_txt_word, drug_name, 20, newpath_1, opt="smallest")

        plt.figure()
        pd.Series((df['sentiment_body'].values)).value_counts().plot.pie(autopct='%.2f', fontsize=10, figsize=(6, 6))
        plt.title(drug_name + "sentiment distribution")
        plt.savefig("%s/%ssentiment_distribution.png" % (newpath_1, drug_name))
        plt.close()

        sentence_lengths = [len(tokens) for tokens in df["selftext_byWords"]]
        VOCAB = sorted(list(set(total_txt_word)))
        print("%s words total, with a vocabulary size of %s" % (len(total_txt_word), len(VOCAB)))
        print("Max sentence length is %s" % max(sentence_lengths))

        fig = plt.figure(figsize=(11, 11))
        plt.xlabel('Sentence length')
        plt.ylabel('Number of sentences')
        plt.hist(sentence_lengths)
        plt.title(drug_name)
        plt.savefig("%s/%ssentence_distribution.png" % (newpath_1, drug_name))
        plt.close()
    return top5_drug_txt_parse, top_5_drug_ment_sub