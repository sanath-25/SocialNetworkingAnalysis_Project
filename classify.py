"""
Classify data.
"""
import pickle
import numpy as np
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from scipy.sparse import lil_matrix
import re
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter

def afinn_sent(terms, afinn, verbose=False):
    postive = 0
    negative = 0
    for term in terms:
        if term in afinn:
            if verbose:
                print('\t%s=%d' % (term, afinn[t]))
            if afinn[term] > 0:
                postive += afinn[term]
            else:
                negative += -1 * afinn[term]
    return postive, negative


def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()

def pn(tokens,tweets,afinn):
    positives = []
    negatives = []
    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sent(token_list, afinn)
        if pos > neg:
            positives.append((tweet['text'], pos, neg))
        elif neg > pos:
            negatives.append((tweet['text'], pos, neg))
    return positives[0],negatives[1]


def posnegs(t,afinn):
    sorted_tokens = tokenize(t['text'])
    pos, neg = afinn_sent(sorted_tokens, afinn)
    if pos > neg:
        return 1
    elif neg > pos:
         return 0
    else:
        return -1
def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))
    for tokens in tokens_list:
        for token in tokens:
             vocabulary[token]
    return vocabulary


def make_feature_matrix(tokens_list, vocabulary,tweets):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()

def do_cross_val(X, y, nfolds):
    
    cv = KFold(n_splits = nfolds, shuffle = False, random_state = 42)
    accuracies = []
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    for train_idx, test_idx in cv.split(X):
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg

def main():
    tweets = pickle.load(open('tweet_64.pkl', 'rb'))
    tweets = [t for t in tweets if 'user' in t]
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    tokens = [tokenize(t['text']) for t in tweets]
    vocabulary = make_vocabulary(tokens)
    y = np.array([posnegs(t,afinn) for t in tweets])
    X = make_feature_matrix(tokens, vocabulary,tweets)
    theta = np.ones(len(vocabulary))
    z = np.zeros(len(tweets))
    for i in range(len(tweets)):
        for j in range(X.indptr[i], X.indptr[i + 1]):
            colidx = X.indices[j]
            z[i] += theta[colidx] * X.data[j]

    print('average accuracy', do_cross_val(X, y, 5))
    k3=[]
    k4=[]
    for l2,k2 in Counter(y).items():
        if l2>=0:
            if(l2==0):
                k3=k2
#                 positives
            else:
                k4=k2
                pickle.dump(k3, open('opclf.pkl', 'wb'))
    pt=[]
    pt1=[]
    r=pn(tokens, tweets, afinn)
    pt.append(r[0])
    pt1.append(r[1])

    pickle.dump((k3, k4,pt,pt1), open('opclf.pkl', 'wb'))



if __name__ == '__main__':
    main()