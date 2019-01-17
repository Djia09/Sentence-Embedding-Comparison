import numpy as np
import pandas as pd
import math
import os
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

def csvToDict(path, columns):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    print("There are %d pairs in %s." % (len(data), path))
    table = [dict(zip(columns, x.split('\t'))) for x in data]
    return table

def loadModel(args):
    # Load model according to the choice in "embed".
    print("Begin loading model...")
    embed = args.embedding
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
            print('Conceptnet model path: ' + mNb_path + '. Warning: in miniNumberbatch case, it must be the FILE.h5 path.')
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
        if args.path:
            inferSent_path = args.path
            print('InferSent model path: ' + inferSent_path + '. Warning: in InferSent case, it must be InferSent FOLDER path.')
        else:
            print('You need to give InferSent FOLDER path')
            sys.exit()
        if args.version == 1 or args.version == 2:
            model_version = args.version
        else:
            print('You need to choose InferSent version between 1 (Word2Vec input) or 2 (FastText input).')
            sys.exit()
        if args.embedding_path:
            W2V_PATH = args.embedding_path
            print('InferSent pretrained embedding path: ' + W2V_PATH + '. Warning: in this case, it must be "model.txt" or "model.vec" path.')
        else:
            print('You need to give InferSent "model.txt" or "model.vec" path path')
            sys.exit()
        MODEL_PATH = os.path.join(inferSent_path, "./encoder/infersent%s.pkl" % model_version)
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))

        # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
        # W2V_PATH = './../../../../Perso/Pretrained-Embedding/GloVe/glove.840B.300d.txt' if model_version == 1 else '../dataset/fastText/crawl-300d-2M.vec'
        model.set_w2v_path(W2V_PATH)
        # Load embeddings of K most frequent words
        vocab_size = 100000
        model.build_vocab_k_words(K=vocab_size)
        d = model.encode(['hello guys']).shape[1]

    print('Model '+embed.upper()+' loaded in %fs.' % (time.time()-start))
    print("Vocabulary size: %d" % vocab_size)
    print("Vector dimension: %d" % d)
    
    return model, d

def vectorize_sentence(sent, model):
	if type(sent) == str:
		sent = sent.lower()
		tokens = word_tokenize(sent)
	elif type(sent) == list:
		tokens = sent
	else:
		print("Wrong input type ! Type: ", type(sent))
		sys.exit()
	d = len(model['hello'])
	vec_list = [model[x] for x in tokens if x in model.keys()]
	if vec_list != []:
		vec_sum = np.array(vec_list).sum(axis=0)
	else:
		vec_sum = np.zeros((d,))
	return vec_sum
	
def getTwoVectors(path, embed, model, save_path1, save_path2):
	# Process and validate
	columns = ["genre", "filename", "year", "ID", "score", "sentence1", "sentence2"]
	table_dev = csvToDict(path, columns)
	df_dev = pd.DataFrame(data=table_dev, columns=columns)
	sent_dev1 = list(df_dev["sentence1"])
	sent_dev2 = list(df_dev["sentence2"])
	assert(len(sent_dev1)==len(sent_dev2))
	# Embedding
	print("Begin embedding...")
	start = time.time()
	if embed == 'elmo':
		X_dev1 = np.array([model.embed_sentence(x.split()).mean(axis=0).mean(axis=0) for x in sent_dev1])
		X_dev2 = np.array([model.embed_sentence(x.split()).mean(axis=0).mean(axis=0) for x in sent_dev2])
	elif embed == 'infersent':
		try:
			X_dev1 = np.load(save_path1)
			X_dev2 = np.load(save_path2)
		except FileNotFoundError:
			X_dev1 = np.zeros((len(sent_dev1), 4096), dtype=np.float32)
			X_dev2 = np.zeros((len(sent_dev2), 4096), dtype=np.float32)
			for i in range(len(sent_dev1)):
				start2 = time.time()
				X_dev1[i,:] = model.encode([sent_dev1[i]], tokenize=True)[0]
				X_dev2[i,:] = model.encode([sent_dev2[i]], tokenize=True)[0]
				print('%d/%d in %fs' % (i, len(sent_dev2), time.time()-start2))
			np.save(save_path1, X_dev1)
			np.save(save_path2, X_dev2)
	else:
		X_dev1 = np.array([vectorize_sentence(x, model) for x in sent_dev1])
		X_dev2 = np.array([vectorize_sentence(x, model) for x in sent_dev2])
	print("Vectorized in %fs" % (time.time()-start))
	print("Shape of X: ", X_dev1.shape)
	return X_dev1, X_dev2

def cosine_similarity(vec1, vec2):
	return np.dot(vec1, vec2.T) / (np.linalg.norm(vec1, 2)*np.linalg.norm(vec2, 2))

def infersent_save_path(dev_test, voc_size, glove_fasttext, version):
	return "./infersent_"+dev_test+"_embedding_"+str(voc_size)+"K_"+glove_fasttext+str(version)+".npy"

def main():
	# Choice of the training and testing task.
	training_path = 'stsbenchmark/sts-train.csv'
	path_dev = 'stsbenchmark/sts-dev.csv'
	path_test = 'stsbenchmark/sts-test.csv'

	# Choice of the embedding model
	parser = argparse.ArgumentParser(description="Comparison of embedding methods for Twitter Sentiment-Analysis.")
	parser.add_argument("-embed", "--embedding", help="Available embedding: glove, miniNumberbatch, elmo, infersent")
	parser.add_argument("-d", "--dimension", help="Choose a dimension for GloVe vectors. Default is the smallest: d=50.", type=int)
	parser.add_argument("-p", "--path", help="Path to your embedding model")
	parser.add_argument("--which_elmo", help="Choose ELMo model weights. Default is the smallest: which_elmo=small.")
	parser.add_argument("-v", "--version", help="Choose InferSent version: 1 or 2.", type=int)
	parser.add_argument("-ep", "--embedding_path", help="For InferSent case, path to your embedding model.")
	args = parser.parse_args()
	embed = args.embedding
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

	# Prepare the output folder
    # if not os.path.exists("Output_Dev"):
	# 	os.makedirs("Output_Dev")
    # if not os.path.exists("Output_Test"):
    #     os.makedirs("Output_Test")
	if embed == "infersent":
		output_path_dev = "./Output_Dev/"+"STS.output.dev."+embed+str(args.version)+".txt"
		output_path_test = "./Output_Test/"+"STS.output.test."+embed+str(args.version)+".txt"
	else:
		output_path_dev = "./Output_Dev/"+"STS.output.dev."+embed+str(d)+".txt"#'withPunctuation.txt'
		output_path_test = "./Output_Test/"+"STS.output.test."+embed+str(d)+".txt"#'withPunctuation.txt'
	# Validation: process, vectorize, evaluate and save.
	print("Import and processing validation data...")
	inf_dev1 = infersent_save_path("dev", 100, "fasttext", 1)
	inf_dev2 = infersent_save_path("dev", 100, "fasttext", 2)
	X_dev1, X_dev2 = getTwoVectors(path_dev, embed, model, inf_dev1, inf_dev2)
	print(X_dev1.shape, X_dev2.shape)
	assert(X_dev1.shape==X_dev2.shape)
	i = 0
	similarity_dev = [str(cosine_similarity(X_dev1[i], X_dev2[i])) for i in range(len(X_dev1))]
	with open(output_path_dev, 'w', encoding='utf-8') as f:
		f.write("\n".join(similarity_dev))
	print('Result saved at %s.' % (output_path_dev))

	# Process and testing
	print("Import and processing testing data...")
	inf_test1 = infersent_save_path("test", 100, "fasttext", 1)
	inf_test2 = infersent_save_path("test", 100, "fasttext", 2)
	X_test1, X_test2 = getTwoVectors(path_test, embed, model, inf_test1, inf_test2)
	assert(len(X_test1)==len(X_test2))
	similarity_test = [str(cosine_similarity(X_test1[i], X_test2[i])) for i in range(len(X_test1))]
	with open(output_path_test, 'w', encoding='utf-8') as f:
		f.write("\n".join(similarity_test))
	print('Result saved at %s.' % (output_path_test))

if __name__ == '__main__':
    main()
