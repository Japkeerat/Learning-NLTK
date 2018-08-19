# Naive Bayes Algorithm implementation for categorizing moview reviews as positive or negative

import random
from nltk.corpus import movie_reviews as mr
from nltk import FreqDist as fd
from nltk import NaiveBayesClassifier as nbc
from nltk import classify
import pickle


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
training_set = featuresets[:2200]
testing_set = featuresets[2200:]
try:
    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
except Exception as e:
    classifier = nbc.train(training_set)

print(classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(10)

save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
