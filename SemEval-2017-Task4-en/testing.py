from training import train, vectorize_sentence
from numberbatch import loadAndCreateNumberBatchModel
from GloVe import loadAndCreateModel
from miniNumberbatch import loadMiniNumberbatch
from preprocessing import getTweetsAndLabels
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from random import randint
import torch
import math
import nltk
from nltk.tokenize import word_tokenize
from allennlp.commands.elmo import ElmoEmbedder

# Choice of the embedding model
embed = "infersent"
dim = 300

# Choice of the training and testing task.
year = str(2016)
subtask = 'A'
training_path = "2017_English_final/GOLD/Subtask_"+subtask+"/twitter-"+year+"train-A.txt"
path_dev = "2017_English_final/GOLD/Subtask_"+subtask+"/twitter-"+year+"dev-A.txt"
path_test = "2017_English_final/GOLD/Subtask_"+subtask+"/twitter-"+year+"test-A.txt"
output_path = "./Output/"+year+"_subtask"+subtask+"_test_english_"+embed+str(dim)+'.txt'

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

def intToLabel(values):
    labels = []
    for x in values:
        if x == 1:
            labels.append("positive")
        elif x == 0:
            labels.append('neutral')
        elif x == -1:
            labels.append("negative")
        else:
            print("Suspiscious value in labels: ", x)
            return x
    return labels

# Load model according to the choice in "embed".
print("Begin loading model...")
start = time.time()
if embed == "glove":
    model = loadAndCreateModel(dim)
    vocab_size = len(model.keys())
    d = len(model['hello'])
elif embed == "numberBatch":
    start = time.time()
    dim = 300
    model = loadAndCreateNumberBatchModel()
    vocab_size = len(model.keys())
    d = len(model['hello'])
elif embed == "miniNumberbatch":
    start = time.time()
    dim = 300
    model = loadMiniNumberbatch()
    vocab_size = len(model.keys())
    d = len(model['hello'])
elif embed == 'elmo':
    ### ELMo embedding on training data
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"

    start = time.time()
    print("Downloading elmo model...")
    model = ElmoEmbedder(options_file, weight_file)
    d = model.embed_sentence(['Hello']).shape[2]
    vocab_size = 0
    print("Downloaded in %fs" % (time.time()-start))
elif embed == 'infersent':
    from models import InferSent
    nltk.download('punkt')
    model_version = 1
    MODEL_PATH = "./../../../../Perso/InferSent/encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = './../../../../Perso/Pretrained-Embedding/GloVe/glove.840B.300d.txt' if model_version == 1 else '../dataset/fastText/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)
    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)
    vocab_size = 100000
    d = 4096

print('Model '+embed.upper()+' loaded in %fs.' % (time.time()-start))
print("Vocabulary size: %d" % vocab_size)
print("Vector dimension: %d" % d)

# Process and training 
grid = False # If grid=True, will perform scikit-learn GridSearch for SVM.
clf = train(training_path, model, grid)

# Process and validate
print("Import and processing validation data...")
tweets_dev, labels_dev, _ = getTweetsAndLabels(path_dev)
y_dev = labelToInt(labels_dev)
print("There are %d tweets and %d labels." % (len(tweets_dev), len(y_dev)))

print("Begin embedding validation...")
start = time.time()
if embed == 'elmo':
    X_dev = np.array([model.embed_sentence(x.split()).mean(axis=0).mean(axis=0) for x in tweets_dev])
elif embed == 'infersent':
    try:
        X_dev = np.load('infersent_validation_embedding.npy')
    except FileNotFoundError: 
        X_dev = model.encode(tweets_dev, tokenize=True)
        np.save('infersent_validation_embedding.npy', X_dev)
else:
    X_dev = np.array([vectorize_sentence(x, model) for x in tweets_dev])
print("Vectorized in %fs" % (time.time()-start))
print("Shape of X: ", X_dev.shape)

print('Begin dev prediction...')
y_pred_dev = clf.predict(X_dev)
print('Dev. accuracy score: ', accuracy_score(y_dev, y_pred_dev))
print('Dev. precision score: ', precision_score(y_dev, y_pred_dev, average='weighted'))
print('Dev. recall score: ', recall_score(y_dev, y_pred_dev, average='weighted'))
print('Dev. f1 score: ', f1_score(y_dev, y_pred_dev, average='weighted'))

# Process and testing
print("Import and processing testing data...")
tweets_test, labels_test, ID = getTweetsAndLabels(path_test)
y_test = labelToInt(labels_test)
print("There are %d tweets and %d labels." % (len(tweets_test), len(labels_test)))
print("Begin embedding testing...")
start = time.time()
if embed == 'elmo':
    X_test = np.array([model.embed_sentence(x.split()).mean(axis=0).mean(axis=0) for x in tweets_test])
elif embed == 'infersent':
    try:
        X_test = np.load('infersent_testing_embedding.npy')
    except FileNotFoundError:
        X_test = np.zeros((len(tweets_test), 4096), dtype=np.float32)
        for i in range(len(tweets_test)):
            start2 = time.time()
            X_test[i,:] = model.encode([tweets_test[i]], tokenize=True)[0]
            print('%d/%d in %fs' % (i, len(tweets_test), time.time()-start2))
        np.save('infersent_testing_embedding.npy', X_test)
else:
    X_test = np.array([vectorize_sentence(x, model) for x in tweets_test])
print("Vectorized in %fs" % (time.time()-start))
print("Shape of X: ", X_test.shape)

print('Begin test prediction...')
start = time.time()
y_pred_test = clf.predict(X_test)
print('Testing accuracy score: ', accuracy_score(y_test, y_pred_test))
print('Testing precision score: ', precision_score(y_test, y_pred_test, average='weighted'))
print('Testing recall score: ', recall_score(y_test, y_pred_test, average='weighted'))
print('Testing f1 score: ', f1_score(y_test, y_pred_test, average='weighted'))
print("Prediction done in %fs." % (time.time()-start))

labels_pred = intToLabel(y_pred_test)
with open(output_path, 'w', encoding='utf-8') as f:
    assert(len(y_pred_test)==len(ID))
    for i in range(len(ID)):
        f.write(str(ID[i]) + '\t' + labels_pred[i] + '\n')
print('Result saved at %s.' % (output_path))
# with open('score.csv','a', encoding='utf-8') as f:
#     f.write(year+';'+subtask+';'+embed+';'+str(dim)+';'+str(accuracy_score(y_test, y_pred_test))+';'+str(precision_score(y_test, y_pred_test, average='weighted'))+';'+str(recall_score(y_test, y_pred_test, average='weighted'))+';'+str(f1_score(y_test, y_pred_test, average='weighted'))+'\n')
