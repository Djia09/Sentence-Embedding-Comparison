from training import train, vectorize_sentence
from GloVe import loadAndCreateModel
from preprocessing import getTweetsAndLabels
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from nltk.tokenize import word_tokenize

path_test = "2017_English_final/GOLD/Subtask_A/twitter-2016test-A.txt"

def labelToInt(labels):
    values = []
    for x in labels:
        if x == "positive":
            values.append(1)
        elif x == 'neutral':
            values.append(0)
        elif x == "negative":
            values.append(-1)
        else:
            print("Suspiscious value in labels: ", x)
            return x
    return values

print("Import and processing testing data...")
tweets_test, labels_test, ID = getTweetsAndLabels(path_test)
y_test = labelToInt(labels_test)
print("There are %d tweets and %d labels." % (len(tweets_test), len(labels_test)))
print("Begin embedding validation...")

