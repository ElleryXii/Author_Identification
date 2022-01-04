# 2019 Nov 28

import pandas
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import log_loss
import xgboost as xgb
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def get_data(filepath):
    data = pandas.read_csv(filepath)
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(data.author.values)
    x_train, x_test, y_train, y_test = train_test_split(data.text.values, y, stratify=y, test_size=0.1, shuffle=True)
    return x_train, x_test, y_train, y_test


def tf_idf(x_train, x_test, language):
    if language == 'en':
        tfv = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
    else:
        tfv = TfidfVectorizer(min_df=3, analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
    tfv.fit(list(x_train) + list(x_test))
    x_train_tfv = tfv.transform(x_train)
    x_test_tfv = tfv.transform(x_test)
    return x_train_tfv, x_test_tfv


def word_count(x_train, x_test, language):
    if language == 'en':
        ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3))
    else:
        ctv = CountVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 3))
    ctv.fit(list(x_train) + list(x_test))
    x_train_ctv = ctv.transform(x_train)
    x_test_ctv = ctv.transform(x_test)
    return x_train_ctv, x_test_ctv


# load word embedding
def get_word_embedding(filepath):
    embeddings = {}
    f = open(filepath, encoding='utf8')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        try:
            coe = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coe
        except ValueError:
            pass
    f.close()
    return embeddings


# Create a vector for the sentence
def sentence_to_vector(sentence, embedding, language):
    if language == 'en':
        words = str(sentence).lower()
        words = word_tokenize(words)
    else:
        words = str(sentence).lower()
        words = [c for c in words]
    M = []
    for w in words:
        try:
            M.append(embedding[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


def word_vector(x_train, x_test, embedding, language):
    x_train_glove = np.array([sentence_to_vector(x, embedding, language) for x in tqdm(x_train)])
    x_test_glove = np.array([sentence_to_vector(x, embedding, language) for x in tqdm(x_test)])
    return x_train_glove, x_test_glove



def xgb_model(x_train, x_test, y_train, y_test, embedding, language):
    x_train_glove, x_test_glove = word_vector(x_train, x_test, embedding, language)
    par = {'min_child_weight': 1, 'eta': 0.1, 'colsample_bytree': 0.7, 'max_depth': 3,
               'subsample': 0.8, 'lambda': 2.0, 'nthread': -1, 'silent': 1,
               'eval_metric': "mlogloss", 'objective': 'multi:softprob', 'num_class': 3}
    dtrain = xgb.DMatrix(x_train_glove, label=y_train)
    watchlist = [(dtrain, 'train')]
    xgb_model = xgb.train(par, dtrain, 1000, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=40)
    dtest = xgb.DMatrix(x_test_glove)
    y_pred = xgb_model.predict(dtest)
    print_analytics(y_test, y_pred, language+'XGBoost')


def fit_classifier(clf, x_train, x_test, y_train, y_test, description=''):
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    print_analytics(y_test, y_pred, description)


def print_analytics(y_test, y_pred, description):
    print(description)
    print("logloss %0.3f " % log_loss(y_test, y_pred))
    y_pred = [np.argmax(p) for p in y_pred]
    print('Accuracy Score :', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def main():
    clf_logreg = LogisticRegression(C=1.0, solver='lbfgs', multi_class='auto', max_iter=250)
    clf_nb = MultinomialNB()
    #
    # # English
    # x_train, x_test, y_train, y_test = get_data("data/en_train.csv")
    #
    # x_train_tfv, x_test_tfv = tf_idf(x_train, x_test, 'en')
    # x_train_ctv, x_test_ctv = word_count(x_train, x_test, 'en')
    #
    # # # tf-idf
    # fit_classifier(clf_logreg, x_train_tfv, x_test_tfv, y_train, y_test, description='English tf-idf logic regression: ')
    # fit_classifier(clf_nb, x_train_tfv, x_test_tfv, y_train, y_test, description='English tf-idf naive_bayes: ')
    #
    # # # word count
    # fit_classifier(clf_logreg, x_train_ctv, x_test_ctv, y_train, y_test, description='English word count logic regression: ')
    # fit_classifier(clf_nb, x_train_ctv, x_test_ctv, y_train, y_test, description='English word count naive_bayes: ')
    #
    # # # Word Embedding
    # embedding = get_word_embedding("data/en_wordembedding.txt")
    # xgb_model(x_train, x_test, y_train, y_test, embedding, 'en')
    # x_train_glove, x_test_glove = word_vector(x_train, x_test, embedding,'en')
    # fit_classifier(clf_logreg, x_train_glove, x_test_glove, y_train, y_test, description='English word embedding logic regression: ')

    # # Chinese
    x_train, x_test, y_train, y_test = get_data("data/cn_train.csv")

    x_train_tfv, x_test_tfv = tf_idf(x_train, x_test, 'cn')
    x_train_ctv, x_test_ctv = word_count(x_train, x_test, 'cn')

    # # tf-idf
    fit_classifier(clf_logreg, x_train_tfv, x_test_tfv, y_train, y_test, description='Chinese tf-idf logic regression: ')
    fit_classifier(clf_nb, x_train_tfv, x_test_tfv, y_train, y_test, description='Chinese tf-idf naive_bayes: ')

    # # word count
    fit_classifier(clf_logreg, x_train_ctv, x_test_ctv, y_train, y_test, description='Chinese word count logic regression: ')
    fit_classifier(clf_nb, x_train_ctv, x_test_ctv, y_train, y_test, description='Chinese word count naive_bayes: ')

    # # Word Embedding
    embedding = get_word_embedding("data/cn_wordembedding.txt")
    xgb_model(x_train, x_test, y_train, y_test, embedding, 'cn')
    x_train_glove, x_test_glove = word_vector(x_train, x_test, embedding, 'cn')
    fit_classifier(clf_logreg, x_train_glove, x_test_glove, y_train, y_test, description='Chinese word embedding logic regression: ')


if __name__ == "__main__":
    main()
