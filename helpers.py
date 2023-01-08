import re
import numpy as np
import nltk  # natural language toolkit
from matplotlib import pyplot as plt
from sklearn import metrics

nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd

from string import punctuation  # solving punctuation problems
from nltk.corpus import stopwords  # stop words in sentences
from pytube import YouTube, Playlist
from sklearn.feature_extraction.text import TfidfVectorizer
from afinn import Afinn


def to_lower(text):
    """
    :param text:
    :return:
        Converted text to lower case as in, converting "Hello" to "hello" or "HELLO" to "hello".
    """
    return text.lower()


def remove_numbers(text):
    """
    :param text:
    :return:
        take string input and return a clean text without numbers.
        Use regex to discard the numbers.
    """
    output = ''.join(c for c in text if not c.isdigit())
    return output


def remove_punct(text):
    """
        :param text:
        :return:
            take string input and return a clean text without punctuation.
    """
    res = ''
    for c in text:
        if c not in punctuation:
            res += c
    return res


def remove_stopwords(sentence):
    """
    removes all the stop words like "is,the,a, etc."
    """
    stop_words = stopwords.words('english')
    return ' '.join([w for w in nltk.word_tokenize(sentence) if w not in stop_words])


def generate_data():
    like_template = r'[0-9]{1,3},?[0-9]{0,3},?[0-9]{0,3} like'
    playlist_link = 'https://www.youtube.com/watch?v=N8UMrI6nC3k&list=PL6XRrncXkMaW5p7muaR2s2IqjouQh4jqS'
    playlist = Playlist(playlist_link)
    urls = playlist.video_urls
    descriptions = []
    links = []
    like_counts = []
    view_counts = []
    ratings = []

    for link in urls[:500]:
        yt = YouTube(link)
        descriptions.append(yt.description)
        links.append(link)
        view_counts.append(yt.views)
        ratings.append(yt.rating)
        str_likes = re.search(like_template, str(yt.initial_data)).group(0)
        like_counts.append(int(str_likes.split(' ')[0].replace(',', '')))

    df = pd.DataFrame(
        {"link": links, "description": descriptions, "likes": like_counts, "views": view_counts, "rating": ratings})

    return df


def write_csv(df):
    df.to_csv("data.csv", index=False)


def correct_descriptions():
    df = pd.read_csv('data.csv', index_col=False)

    documents = df['description']

    for i, document in enumerate(documents):
        documents[i] = remove_punct(document)
        documents[i] = remove_stopwords(documents[i])
        documents[i] = remove_numbers(documents[i])
        documents[i] = to_lower(documents[i])

    df['description'] = documents
    df.to_csv("cleaned_data.csv", index=False)


def calculate_tf_idf(corpus):
    tr_idf_model = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(corpus)
    tf_idf_array = tf_idf_vector.toarray()

    words_set = tr_idf_model.get_feature_names()
    df_tf_idf = pd.DataFrame(tf_idf_array, columns=words_set)

    return df_tf_idf


def save_tf_idf(df):
    df.to_csv("tf_idf.csv", index=False)


def save_sentiments():
    afn = Afinn()
    df = pd.read_csv("cleaned_data.csv", index_col=False)

    documents = df['description']

    scores = [afn.score(document) for document in documents]

    sentiments = ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' for score in scores]

    df['score'] = scores
    df['sentiment'] = sentiments

    df.to_csv("data_with_sentiment_analysis.csv", index=False)


def efron_rsquare(y, y_pred):
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)


def full_log_likelihood(w, X, y):
    score = np.dot(X.reshape((-1, 1)), w.reshape((1, -1)))
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)


def null_log_likelihood(w, X, y):
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)


def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))


def mcfadden_adjusted_rsquare(w, X, y):
    k = len(X)
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))


def plot_roc(model, X_test, y_test):
    # calculate the fpr and tpr for all thresholds of the classification
    probabilities = model.predict_proba(np.array(X_test))
    predictions = probabilities[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
