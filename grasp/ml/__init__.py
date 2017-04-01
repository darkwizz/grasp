# ---- MODEL --------------------------------------------------------------------------------------
# The Model base class is inherited by Perceptron, Bayes, ...
import collections
import json
import random

import math

import re

import itertools

from grasp.iteration import shuffled


class Model(object):
    def __init__(self, examples=[], **kwargs):
        self.labels = {}

    def train(self, v, label=None):
        raise NotImplementedError

    def predict(self, v):
        raise NotImplementedError

    def save(self, f):
        json.dump(self.__dict__, f)

    @classmethod
    def load(cls, f):
        self = cls()
        for k, v in json.load(f).items():
            try:
                getattr(self, k).update(v)  # defaultdict?
            except:
                setattr(self, k, v)
        return self


# ---- PERCEPTRON ----------------------------------------------------------------------------------
# The Perceptron or single-layer neural network is a supervised machine learning algorithm.
# Supervised machine learning uses labeled training examples to infer statistical patterns.
# Each example is a set of features – e.g., set(('lottery',)) – and a label (e.g., 'spam').

# The Perceptron takes a list of examples and learns what features are associated with what labels.
# The resulting 'model' can then be used to predict the label of new examples.

def avg(a):
    a = list(a)
    n = len(a) or 1
    s = sum(a)
    return float(s) / n


def sd(a):
    a = list(a)
    n = len(a) or 1
    m = avg(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / n)


def iavg(x, m=0.0, sd=0.0, t=0):
    """ Returns the iterative (mean, standard deviation, number of samples).
    """
    t += 1
    v = sd ** 2 + m ** 2  # variance
    v += (x ** 2 - v) / t
    m += (x ** 1 - m) / t
    sd = math.sqrt(v - m ** 2)
    return (m, sd, t)


# p = iavg(1)     # (1.0, 0.0, 1)
# p = iavg(2, *p) # (1.5, 0.5, 2)
# p = iavg(3, *p) # (2.0, 0.8, 3)
# p = iavg(4, *p) # (2.5, 1.1, 4)
#
# print(p)
# print(sum([1,2,3,4]) / 4.)

def softmax(p, a=1.0):
    """ Returns a dict with float values that sum to 1.0
        (using generalized logistic regression).
    """
    if p:
        a = a or 1
        v = p.values()
        v = [x / a for x in v]
        m = max(v)
        e = [math.exp(x - m) for x in v]  # prevent overflow
        s = sum(e)
        v = [x / s for x in e]
        p = dict(zip(p.keys(), v))
    return p


# print(softmax({'cat': +1, 'dog': -1})) # {'cat': 0.88, 'dog': 0.12}
# print(softmax({'cat': +2, 'dog': -2})) # {'cat': 0.98, 'dog': 0.02}

def top(p):
    """ Returns a (key, value)-tuple with the max value in the dict.
    """
    if p:
        v = max(p.values())
    else:
        v = 0.0
    k = [k for k in p if p[k] == v]
    k = random.choice(k)
    return k, v


# print(top({'cat': 1, 'dog': 2})) # ('dog', 2)

class Perceptron(Model):
    def __init__(self, examples=[], n=10, **kwargs):
        """ Single-layer averaged perceptron learning algorithm.
        """
        # {label: count}
        # {label: {feature: (weight, weight sum, timestamp)}}
        self.labels = collections.defaultdict(int)
        self.weights = collections.defaultdict(dict)

        self._t = 1
        self._p = iavg(0)

        for i in range(n):
            for v, label in shuffled(examples):
                self.train(v, label)

    def train(self, v, label=None):

        def cumsum(label, f, i, t):
            # Accumulate average weights (prevents overfitting).
            # Keep running sum + time when sum was last updated.
            # http://www.ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf
            w = self.weights[label].setdefault(f, [0, 0, 0])
            w[0] += i
            w[1] += w[0] * (t - w[2])
            w[2] = t

        self.labels[label] += 1

        guess, p = top(self.predict(v, normalize=False))
        if guess != label:
            for f in v:
                # Error correction:
                cumsum(label, f, +1, self._t)
                cumsum(guess, f, -1, self._t)
            self._t += 1

        self._p = iavg(abs(p), *self._p)  # (mean, sd, t)

    def predict(self, v, normalize=True):
        """ Returns a dict of (label, probability)-items.
        """
        p = dict.fromkeys(self.labels, 0.0)
        t = float(self._t)
        for label, features in self.weights.items():
            n = 0
            for f in v:
                if f in features:
                    w = features[f]
                    n = n + (w[1] + w[0] * (t - w[2])) / t
            p[label] = n

        if normalize:
            # 1. Divide values by avg + sd (-1 => +1)
            # 2. Softmax to values between 0.1 => 0.9
            #    (with < 0.1 and > 0.9 for outliers)
            p = softmax(p, a=(self._p[0] + self._p[1]))
        return p


# p = Perceptron(examples=[
#     (('woof', 'bark'), 'dog'),
#     (('meow', 'purr'), 'cat')], n=10)
#
# print(p.predict(('meow',)))
#
# p.save(open('model.json', 'w'))
# p = Perceptron.load(open('model.json'))

# ---- NAIVE BAYES ---------------------------------------------------------------------------------
# The Naive Bayes model is a simple alternative for Perceptron (it trains very fast).
# It is based on the likelihood that a given feature occurs with a given label.

# The probability that something big and bad is a wolf is:
# p(big|wolf) * p(bad|wolf) * p(wolf) / (p(big) * p(bad)).

# So it depends on the frequency of big wolves, bad wolves,
# other wolves, other big things, and other bad things.

class Bayes(Model):
    def __init__(self, examples=[], **kwargs):
        """ Binomial Naive Bayes learning algorithm.
        """
        # {label: count}
        # {label: {feature: count}}
        self.labels = collections.defaultdict(int)
        self.weights = collections.defaultdict(dict)

        for v, label in examples:
            self.train(v, label)

    def train(self, v, label=None):
        for f in v:
            try:
                self.weights[label][f] += 1
            except KeyError:
                self.weights[label][f] = 1 + 0.1  # smoothing
        self.labels[label] += 1

    def predict(self, v):
        """ Returns a dict of (label, probability)-items.
        """
        p = dict.fromkeys(self.labels, 0.0)
        for x in self.labels:
            n = self.labels[x]
            w = (self.weights[x].get(f, 0.1) / n for f in v)
            w = map(math.log, w)  # prevent underflow
            w = sum(w)
            w = math.exp(w)
            w = w * n
            w = w / sum(self.labels.values())
            p[x] = w

        s = sum(p.values()) or 1
        for label in p:
            p[label] /= s
        return p


# ---- FEATURES ------------------------------------------------------------------------------------
# Character 3-grams are sequences of 3 successive characters: 'hello' => 'hel', 'ell', 'llo'.
# Character 3-grams are useful as training examples for text classifiers,
# capturing 'small words' such as pronouns, smileys, word suffixes (-ing)
# and language-specific letter combinations (oeu, sch, tch, ...)

def chngrams(s, n=3):
    """ Returns an iterator of character n-grams.
    """
    for i in range(len(s) - n + 1):
        yield s[i:i + n]  # 'hello' => 'hel', 'ell', 'llo'


def ngrams(s, n=2):
    """ Returns an iterator of word n-grams.
    """
    for w in chngrams([w for w in re.split(r'\s+', s) if w], n):
        yield tuple(w)


def v(s, features=('ch3',)):  # (vector)
    """ Returns a set of character trigrams in the given string.
        Can be used as Perceptron.train(v(s)) or predict(v(s)).
    """
    # s = s.lower()
    s = re.sub(r'(https?://[^\s]+)', 'http://', s)
    s = re.sub(r'(@[\w.]+)', '@me', s, flags=re.U)
    v = collections.Counter()
    v[''] = 1  # bias
    for f in features:
        if f[0] == 'c':  # 'c1' (punctuation, diacritics)
            v.update(chngrams(s, n=int(f[-1])))
        if f[0] == 'w':  # 'w1'
            v.update(ngrams(s, n=int(f[-1])))
    return v


vec = v


# data = []
# for id, username, tweet, date in csv(cd('spam.csv')):
#     data.append((v(tweet), 'spam'))
# for id, username, tweet, date in csv(cd('real.csv')):
#     data.append((v(tweet), 'real'))
#
# p = Perceptron(examples=data, n=10)
# p.save(open('spam-model.json', 'w'))
#
# print(p.predict(v('Be Lazy and Earn $3000 per Week'))) # {'real': 0.15, 'spam': 0.85}

# ---- FEATURE SELECTION ---------------------------------------------------------------------------
# Feature selection identifies the best features, by evaluating their statistical significance.

def pp(data=[]):  # (posterior probability)
    """ Returns a {feature: {label: frequency}} dict
        for the given set of (vector, label)-tuples.
    """
    f1 = collections.defaultdict(float)  # {label: count}
    f2 = collections.defaultdict(float)  # {feature: count}
    f3 = collections.defaultdict(float)  # {feature, label: count}
    p = {}
    for v, label in data:
        f1[label] += 1
    for v, label in data:
        for f in v:
            f2[f] += 1
            f3[f, label] += 1 / f1[label]
    for label in f1:
        for f in f2:
            p.setdefault(f, {})[label] = f1[label] / f2[f] * f3[f, label]
    return p


def fsel(data=[]):  # (feature selection, using chi2)
    """ Returns a {feature: p-value} dict
        for the given set of (vector, label)-tuples.
    """
    from scipy.stats import chi2_contingency as chi2

    f1 = collections.defaultdict(float)  # {label: count}
    f2 = collections.defaultdict(float)  # {feature: count}
    f3 = collections.defaultdict(float)  # {feature, label: count}
    p = {}
    for v, label in data:
        f1[label] += 1
    for v, label in data:
        for f in v:
            f2[f] += 1
            f3[f, label] += 1
    for f in f2:
        p[f] = chi2([[f1[label] - f3[f, label] or 0.1 for label in f1],
                     [f3[f, label] or 0.1 for label in f1]])[1]
    return p


def topn(p, n=10, reverse=False):
    """ Returns an iterator of (key, value)-tuples
        ordered by the highest values in the dict.
    """
    for k in sorted(p, key=p.get, reverse=not reverse)[:n]:
        yield k, p[k]


# data = [
#     (set(('yawn', 'meow')), 'cat'),
#     (set(('yawn',       )), 'dog')] * 10
#
# bias = pp(data)
#
# for f, p in fsel(data).items():
#     if p < 0.01:
#         print(f)
#         print(top(bias[f]))
#         # 'meow' is significant (always predicts 'cat')
#         # 'yawn' is independent (50/50 'dog' and 'cat')

# ---- ACCURACY ------------------------------------------------------------------------------------
# Predicted labels will often include false positives and false negatives.
# A false positive is a real e-mail that is labeled as spam (for example).
# A false negative is a real e-mail that ends up in the junk folder.

# To evaluate how well a model deals with false positives and negatives (i.e., accuracy),
# we can use a list of labeled test examples and check the label vs. the predicted label.
# The evaluation will yield two scores between 0.0 and 1.0: precision (P) and recall (R).
# Higher precision = less false positives.
# Higher recall    = less false negatives.

# A robust evaluation of P/R is by K-fold cross-validation.
# K-fold cross-validation takes a list of labeled examples,
# and trains + tests K different models on different subsets of examples.

def confusion_matrix(model, test=[]):
    """ Returns the matrix of labels x predicted labels, as a dict.
    """
    # { label: { predicted label: count}}
    m = collections.defaultdict(lambda: \
                                    collections.defaultdict(int))
    for label in model.labels:
        m[label]
    for v, label in test:
        guess, p = top(model.predict(v))
        m[label][guess] += 1
    return m


def test(model, target, data=[]):
    """ Returns a (precision, recall)-tuple for the test data.
        High precision = few false positives for target label.
        High recall    = few false negatives for target label.
    """
    if isinstance(model, Model):
        m = confusion_matrix(model, data)
    if isinstance(model, dict):  # confusion matrix
        m = model

    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for x1 in m:
        for x2, n in m[x1].items():
            if target == x1 == x2:
                TP += n
            if target != x1 == x2:
                TN += n
            if target == x1 != x2:
                FN += n
            if target == x2 != x1:
                FP += n
    return (
        TP / (TP + FP or 1),
        TP / (TP + FN or 1))


def kfoldcv(Model, data=[], k=10, weighted=False, debug=False, **kwargs):
    """ Returns the average precision & recall across labels, in k tests.
    """

    def folds(a, k=10):
        # folds([1,2,3,4,5,6], k=2) => [1,2,3], [4,5,6]
        return (a[i::k] for i in range(k))

    def wavg(a):
        # wavg([(1, 0.33), (2, 0.33), (3, 0.33)]) => 2  (weighted mean)
        return sum(v * w for v, w in a) / (sum(w for v, w in a) or 1)

    data = list(shuffled(data))
    data = list(folds(data, k))

    P = []
    R = []
    for i in range(k):
        x = data[i]
        y = data[:i] + data[i + 1:]
        y = itertools.chain(*y)
        m = Model(examples=y, **kwargs)
        f = confusion_matrix(m, test=x)
        for label, n in m.labels.items():
            if not weighted:
                n = 1
            precision, recall = test(f, target=label)
            P.append((precision, n))
            R.append((recall, n))

            if debug:
                # k 1 P 0.99 R 0.99 spam
                print('k %i' % (i + 1), 'P %.2f' % precision, 'R %.2f' % recall, label)

    return wavg(P), wavg(R)


def F1(P, R):
    """ Returns the harmonic mean of precision and recall.
    """
    return 2.0 * P * R / (P + R or 1)


# data = []
# for id, username, tweet, date in csv(cd('spam.csv')):
#     data.append((v(tweet), 'spam'))
# for id, username, tweet, date in csv(cd('real.csv')):
#     data.append((v(tweet), 'real'))
#
# print(kfoldcv(Perceptron, data, k=3, n=5, debug=True)) # ~ P 0.80 R 0.80

# ---- CONFIDENCE ----------------------------------------------------------------------------------
# Predicted labels usually come with a probability or confidence score.
# However, the raw scores of Perceptron + softmax, SVM and Naive Bayes
# do not always yield good estimates of true probabilities.

# We can use a number of training examples for calibration.
# Isotonic regression yields a function that can be used to
# map the raw scores to well-calibrated probabilities.

def pav(y=[]):
    """ Returns the isotonic regression of y
        (Pool Adjacent Violators algorithm).
    """
    y = list(y)
    n = len(y) - 1
    while 1:
        e = 0
        i = 0
        while i < n:
            j = i
            while j < n and y[j] >= y[j + 1]:
                j += 1
            if y[i] != y[j]:
                r = y[i:j + 1]
                y[i:j + 1] = [float(sum(r)) / len(r)] * len(r)
                e = 1
            i = j + 1
        if not e:  # converged?
            break
    return y


# Example from Fawcett & Niculescu (2007), PAV and the ROC convex hull:
#
# y = sorted((
#     (0.90, 1),
#     (0.80, 1),
#     (0.70, 0),
#     (0.60, 1),
#     (0.55, 1),
#     (0.50, 1),
#     (0.45, 0),
#     (0.40, 1),
#     (0.35, 1),
#     (0.30, 0),
#     (0.27, 1),
#     (0.20, 0),
#     (0.18, 0),
#     (0.10, 1),
#     (0.02, 0)
# ))
# y = zip(*y)
# y = list(y)[1]
# print(pav(y))

class calibrate(Model):
    def __init__(self, model, label, data=[]):
        """ Returns a new Model calibrated on the given data,
            which is a set of (vector, label)-tuples.
        """
        self._model = model
        self._label = label
        # Isotonic regression:
        y = ((model.predict(v)[label], label == x) for v, x in data)
        y = sorted(y)  # monotonic
        y = zip(*y)
        y = list(y or ((), ()))
        x = list(y[0])
        y = list(y[1])
        y = pav(y)
        x = [0] + x + [1]
        y = [0] + y + [1]
        f = {}
        i = 0
        # Linear interpolation:
        for p in range(100 + 1):
            p *= 0.01
            while x[i] < p:
                i += 1
            f[p] = (y[i - 1] * (x[i] - p) + y[i] * (p - x[i - 1])) / (x[i] - x[i - 1])
        self._f = f

    def predict(self, v):
        """ Returns the label's calibrated probability (0.0-1.0).
        """
        p = self._model.predict(v)[self._label]
        p = self._f[round(p, 2)]
        return p

    def save(self, f):
        raise NotImplementedError

    @classmethod
    def load(cls, f):
        raise NotImplementedError


# data = []
# for review, polarity in csv('reviews-assorted1000.csv'):
#     data.append((v(review), polarity))
#
# m = Model.load('sentiment.json')
# m = calibrate(m, '+', data)

# ---- VECTOR --------------------------------------------------------------------------------------
# A vector is a {feature: weight} dict, with n features, or n dimensions.

# If {'x1':1, 'y1':2} and {'x2':3, 'y2':4} are two points in 2D,
# then their distance is: sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2).
# The distance can be calculated for points in 3D, 4D, or in nD.

# Another measure of similarity is the angle between two vectors (cosine).
# This works well for text features.

def distance(v1, v2):
    """ Returns the distance of the given vectors.
    """
    return sum(pow(v1.get(f, 0.0) - v2.get(f, 0.0), 2) for f in features((v1, v2))) ** 0.5


def dot(v1, v2):
    """ Returns the dot product of the given vectors.
        Each vector is a dict of (feature, weight)-items.
    """
    return sum(v1.get(f, 0.0) * w for f, w in v2.items())


def norm(v):
    """ Returns the norm of the given vector.
    """
    return sum(w ** 2 for f, w in v.items()) ** 0.5


def cos(v1, v2):
    """ Returns the angle of the given vectors (0.0-1.0).
    """
    return 1 - dot(v1, v2) / (norm(v1) * norm(v2) or 1)  # cosine distance


def knn(v, vectors=[], k=3, distance=cos):
    """ Returns the k nearest neighbors from the given list of vectors.
    """
    nn = sorted((1 - distance(v, x), x) for x in vectors)
    nn = reversed(nn)
    nn = list(nn)[:k]
    return nn


def sparse(v):
    """ Returns a vector with non-zero weight features.
    """
    return v.__class__({f: w for f, w in v.items() if w != 0})


def tf(v):
    """ Returns a vector with normalized weights
        (term frequency, sum to 1.0).
    """
    n = sum(v.values())
    n = float(n or 1)
    return v.__class__({f: w / n for f, w in v.items()})


def tfidf(vectors=[]):
    """ Returns an iterator of vectors with normalized weights
        (term frequency–inverse document frequency).
    """
    df = collections.Counter()  # stopwords have higher df (the, or, I, ...)
    if not isinstance(vectors, list):
        vectors = list(vectors)
    for v in vectors:
        df.update(v)
    for v in vectors:
        yield v.__class__({f: w / float(df[f] or 1) for f, w in v.items()})


def features(vectors=[]):
    """ Returns the set of features for all vectors.
    """
    return set(itertools.chain(*(vectors)))


def centroid(vectors=[]):
    """ Returns the mean vector for all vectors.
    """
    v = list(vectors)
    n = float(len(v))
    return {f: sum(v.get(f, 0) for v in v) / n for f in features(v)}


def majority(a, default=None):
    """ Returns the most frequent item in the given list (majority vote).
    """
    f = collections.Counter(a)
    try:
        m = max(f.values())
        return random.choice([k for k, v in f.items() if v == m])
    except:
        return default


# print(majority(['cat', 'cat', 'dog']))

# examples = [
#     ("'I know some good games we could play,' said the cat.", 'seuss'),
#     ("'I know some new tricks,' said the cat in the hat."   , 'seuss'),
#     ("They roared their terrible roars"                     , 'sendak'),
#     ("They gnashed their terrible teeth"                    , 'sendak'),
#
# ]
#
# v, labels = zip(*examples) # = unzip
# v = list(wc(tokenize(v)) for v in v)
# v = list(tfidf(v))
#
# labels = {id(v): label for v, label in zip(v, labels)} # { vector id: label }
#
# x = wc(tokenize('They rolled their terrible eyes'))
# x = wc(tokenize("'Look at me! Look at me now!' said the cat."))
#
# for w, nn in knn(x, v, k=3):
#     w = round(w, 2)
#     print(w, labels[id(nn)])
#
# print(majority(labels[id(nn)] for w, nn in knn(x, v, k=3)))

# ---- VECTOR CLUSTERING ---------------------------------------------------------------------------
# The k-means clustering algorithm is an unsupervised machine learning method
# that partitions a given set of vectors into k clusters, so that each vector
# belongs to the cluster with the nearest center (mean).

euclidean = distance
spherical = cos


def ss(vectors=[], distance=euclidean):
    """ Returns the sum of squared distances to the center (variance).
    """
    v = list(vectors)
    c = centroid(v)
    return sum(distance(v, c) ** 2 for v in v)


def kmeans(vectors=[], k=3, distance=euclidean, iterations=100, n=10):
    """ Returns a list of k lists of vectors, clustered by distance.
    """
    vectors = list(vectors)
    optimum = None

    for _ in range(max(n, 1)):

        # Random initialization:
        g = list(shuffled(vectors))
        g = list(g[i::k] for i in range(k))[:len(g)]

        # Lloyd's algorithm:
        for _ in range(iterations):
            m = [centroid(v) for v in g]
            e = []
            for m1, g1 in zip(m, g):
                for v in g1:
                    d1 = distance(v, m1)
                    d2, g2 = min((distance(v, m2), g2) for m2, g2 in zip(m, g))
                    if d2 < d1:
                        e.append((g1, g2, v))  # move to nearer centroid
            for g1, g2, v in e:
                g1.remove(v)
                g2.append(v)
            if not e:  # converged?
                break

        # Optimal solution = lowest within-cluster sum of squares:
        optimum = min(optimum or g, g, key=lambda g: sum(ss(g, distance) for g in g))
    return optimum

# data = [
#     {'woof': 1},
#     {'woof': 1},
#     {'meow': 1}
# ]
#
# for cluster in kmeans(data, k=2):
#     print(cluster) # cats vs dogs
