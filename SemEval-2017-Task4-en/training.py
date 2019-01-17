from preprocessing import getTweetsAndLabels
import time
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from allennlp.commands.elmo import ElmoEmbedder
from models import InferSent

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.0001, 0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print("Best parameters: ", grid_search.best_params_)
    return grid_search

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

def vectorize_sentence(sent, model):
    d = len(model['hello'])
    vec = np.zeros((d,))
    count = 0
    for x in word_tokenize(sent):
        if x in model:
            new_vec = model[x]
            count += 1
        else:
            new_vec = np.zeros((d,))
        vec = vec + new_vec
    if count == 0:
        return vec
    else:
        return vec/count

def train(path, model, grid=False):
    tweets, labels, _ = getTweetsAndLabels(path)
    y_train = labelToInt(labels)
    print("There are %d tweets and %d labels." % (len(tweets), len(y_train)))

    print("Begin embedding...")
    start = time.time()
    if isinstance(model, InferSent):
        try:
            X = np.load('infersent_training_embedding.npy')
        except FileNotFoundError:             
            X = np.zeros((len(tweets), 4096), dtype=np.float32)
            for i in range(len(tweets)):
                start2 = time.time()
                X[i,:] = model.encode([tweets[i]], tokenize=True)[0]
                print('%d/%d in %fs' % (i, len(tweets), time.time()-start2))
            np.save('infersent_training_embedding.npy', X)
    elif isinstance(model, ElmoEmbedder):
        X = np.array([model.embed_sentence(x.split()).mean(axis=0).mean(axis=0) for x in tweets])
    else:
        X = np.array([vectorize_sentence(x, model) for x in tweets])
    print("Vectorized in %fs" % (time.time()-start))
    print("Shape of X: ", X.shape)

    print('Begin classification...')
    start = time.time()
    if grid:
        nfolds = 10
        clf = svc_param_selection(X, y_train, nfolds)
        print("Grid Search done for nfolds=%d in %fs." % (nfolds, time.time()-start))
    else:
        clf = SVC(kernel='linear', C=1, gamma=0.001)
        clf.fit(X, y_train)
    print("Training done in %fs" % (time.time()-start))
    return clf
