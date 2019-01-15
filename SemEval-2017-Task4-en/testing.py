import numpy as np
import math
import argparse
import sys
import time
import torch
import nltk
from random import randint
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from training import train, vectorize_sentence
from preprocessing import getTweetsAndLabels

def loadModel(args):
    # Load model according to the choice in "embed".
    print("Begin loading model...")
    embed = args.embed
    start = time.time()
    if embed == "glove":
        from GloVe import loadAndCreateModel
        if args.path:
            path_to_glove = args.path# "./../../../../Perso/Pretrained-Embedding/GloVe/"
            print('GloVe path: ' + path_to_glove + '.\nWarning: in GloVe case, it must be the FOLDER path.')
        else:
            print('You need to give GloVe FOLDER path')
            sys.exit()
        if args.dimension:
            dim = args.dimension
            if dim not in [50, 100, 200, 300]:
                print("Available GloVe dimension: 50, 100, 200 or 300. You chose %d !" % (dim))
                sys.exit()
        else:
            dim = 50
        print('Chosen dimension for GloVe: ', dim)
        start = time.time()
        model = loadAndCreateModel(dim, path_to_glove)
        vocab_size = len(model.keys())
        d = len(model['hello'])
    elif embed == "numberBatch":
        from numberbatch import loadAndCreateNumberBatchModel
        start = time.time()
        dim = 300
        model = loadAndCreateNumberBatchModel()
        vocab_size = len(model.keys())
        d = len(model['hello'])
    elif embed == "miniNumberbatch":
        from miniNumberbatch import loadMiniNumberbatch
        if args.path:
            mNb_path = args.path #"./../17.06/mini.h5"
            print('Conceptnet model path: ' + path_to_w2v + '. Warning: in miniNumberbatch case, it must be the FILE.h5 path.')
        else:
            print('You need to give ConceptNet miniNumberBatch FILE.h5 path')
            sys.exit()
        start = time.time()
        model = loadMiniNumberbatch(mNb_path)
        vocab_size = len(model.keys())
        d = len(model['hello'])
    elif embed == "elmo":
        from allennlp.commands.elmo import ElmoEmbedder

        ### ELMo embedding on training data
        if args.which_elmo:
            which_elmo = args.which_elmo# "small"
            print("Chosen ELMo option: ", which_elmo)
        else:
            which_elmo = "small"
            print('Default ELMo chosen: small.')
        if which_elmo == "small":
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        elif which_elmo == "medium":
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
        elif which_elmo == "original":
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        else:
            print('This option is not available...')
            sys.exit()
        start = time.time()
        print("Downloading elmo model...")
        model = ElmoEmbedder(options_file, weight_file)
        dim = model.embed_sentence(['Hello']).shape[2]
        d = dim
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
        d = model.encode(['hello guys']).shape[1]

    print('Model '+embed.upper()+' loaded in %fs.' % (time.time()-start))
    print("Vocabulary size: %d" % vocab_size)
    print("Vector dimension: %d" % d)
    
    return model, d

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

def validation(path_dev, embed, model, clf):
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

def testing(path_test, embed, model, clf):
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
    return y_pred_test

def main():
    # Choice of the training and testing task.
    year = str(2016)
    subtask = 'A'
    training_path = "2017_English_final/GOLD/Subtask_"+subtask+"/twitter-"+year+"train-A.txt"
    path_dev = "2017_English_final/GOLD/Subtask_"+subtask+"/twitter-"+year+"dev-A.txt"
    path_test = "2017_English_final/GOLD/Subtask_"+subtask+"/twitter-"+year+"test-A.txt"

    # Choice of the embedding model
    parser = argparse.ArgumentParser(description="Comparison of embedding methods for Twitter Sentiment-Analysis.")
    parser.add_argument("--embed", help="Available embedding: glove, miniNumberbatch, elmo, infersent")
    parser.add_argument("-d", "--dimension", help="Choose a dimension for GloVe vectors. Default is the smallest: d=50.", type=int)
    parser.add_argument("-p", "--path", help="Path to your embedding model")
    parser.add_argument("--which_elmo", help="Choose ELMo model weights. Default is the smallest: which_elmo=small.")
    args = parser.parse_args()
    embed = args.embed
    if not embed:
        print('No embed argument chosen, argument "embed": %s. Exit program.' % (embed))
        sys.exit()
    elif embed not in ['glove', 'w2v', 'numberBatch', 'miniNumberbatch', 'elmo', 'infersent']:
        print("Wrong chosen argument 'embed'. Choose between: glove, w2v, miniNumberbatch, elmo")
        sys.exit()
    else:
        print('Chosen embed argument: ', embed)

    # Loading the embedding model
    model, d = loadModel(args)
    output_path = "./Output/"+year+"_subtask"+subtask+"_test_english_"+embed+str(d)+".txt"#'withPunctuation.txt'

    # Process and training 
    grid = False # If grid=True, will perform scikit-learn GridSearch for SVM.
    clf = train(training_path, model, grid)

    # Process and validate
    validation(path_dev, embed, model, clf)

    # Process and testing
    y_pred_test = testing(path_test, embed, model, clf)

    # Converting and saving the result
    labels_pred = intToLabel(y_pred_test)
    with open(output_path, 'w', encoding='utf-8') as f:
        assert(len(y_pred_test)==len(ID))
        for i in range(len(ID)):
            f.write(str(ID[i]) + '\t' + labels_pred[i] + '\n')
    print('Result saved at %s.' % (output_path))

if __name__ == '__main__':
    main()