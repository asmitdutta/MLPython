#!/usr/bin/python

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from scipy.stats import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        print mode(votes)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf

documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

words_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in words_features:
        features[w] = (w in words)
    return features

print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))

featuresets = [(find_features(rev),category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]



classifier = nltk.NaiveBayesClassifier.train(training_set)
#classifier_f = open("naivebayes.pickle","rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()
print("Naive Bayes accuracy",nltk.classify.accuracy(classifier, testing_set) * 100)
classifier.show_most_informative_features(15)

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(training_set)
print("logistic regression accuracy:", nltk.classify.accuracy(logistic_regression_classifier, testing_set))

voted_classifier = VoteClassifier(classifier, logistic_regression_classifier)
print("voted_classifier accuracy percent:",nltk.classify.accuracy(voted_classifier, testing_set)*100)
print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0]))
