# Naive Bayes Algorithm implementation for categorizing moview reviews as positive or negative

import random

from nltk.corpus import movie_reviews as mr
from nltk import FreqDist as fd
from nltk import NaiveBayesClassifier as nbc
from nltk import classify
from nltk.classify.scikitlearn import SklearnClassifier

import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


documents = [(list(mr.words(fileid)), category)
             for category in mr.categories()
             for fileid in mr.fileids(category)]

random.shuffle(documents)

all_words = []
for w in mr.words():
    all_words.append(w.lower())

all_words = fd(all_words)
word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = {w in words}

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]
training_set = featuresets[:2500]
testing_set = featuresets[2500:]
try:
    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
except Exception as e:
    classifier = nbc.train(training_set)

print("Original NaiveBayes: ", classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(10)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB: ", (classify.accuracy(MNB_classifier, testing_set) * 100))

LR = SklearnClassifier(LogisticRegression())
LR.train(training_set)
print("Linear Regression: ", (classify.accuracy(LR, testing_set) * 100))

SGDC = SklearnClassifier(SGDClassifier())
SGDC.train(training_set)
print("SGD Classifier: ", (classify.accuracy(SGDC, testing_set) * 100))

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC: ", (classify.accuracy(SVC_classifier, testing_set) * 100))

LSVC_classifier = SklearnClassifier(LinearSVC())
LSVC_classifier.train(training_set)
print("BernoulliNB: ", (classify.accuracy(LSVC_classifier, testing_set) * 100))

NSVC_classifier = SklearnClassifier(NuSVC())
NSVC_classifier.train(training_set)
print("BernoulliNB: ", (classify.accuracy(NSVC_classifier, testing_set) * 100))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB: ", (classify.accuracy(BNB_classifier, testing_set) * 100))
