#!/usr/bin/env python

import numpy as np
import os
import time
import io
import math
from gensim.models import KeyedVectors
from GloVe import loadAndCreateModel
from miniNumberbatch import loadMiniNumberbatch
from allennlp.commands.elmo import ElmoEmbedder

embed = "elmo"
which_elmo = "original"
print("Begin loading model...")
if embed == "glove":
    start = time.time()
    dim = 50
    model = loadAndCreateModel(dim)
    vocab_size = len(model.keys())
    d = len(model['hello'])

elif embed == "w2v":
    model = KeyedVectors.load_word2vec_format('./../../../Perso/Pretrained-Embedding/Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)
    vocab_size = len(model.vocab)
    d = len(model['hello'])

elif embed == "miniNumberbatch":
    start = time.time()
    dim = 300
    model = loadMiniNumberbatch()
    vocab_size = len(model.keys())
    d = len(model['hello'])

elif embed == "elmo":
    ### ELMo embedding on training data
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
    start = time.time()
    print("Downloading elmo model...")
    model = ElmoEmbedder(options_file, weight_file)
    dim = model.embed_sentence(['Hello']).shape[2]
    d = dim
    vocab_size = 0
    print("Downloaded in %fs" % (time.time()-start))

else:
    print("This embedding method is not available.", embed)
# d = len(model['hello'])
print('Model '+embed.upper()+' loaded in %fs.' % (time.time()-start))
print("Vocabulary size: %d" % vocab_size)
print("Vector dimension: %d" % d)

def cosine_similarity_single(a, b):
    assert len(a)==len(b)
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def getPairWords(which):
    with open("SemEval17-Task2/"+which+"/subtask1-monolingual/data/en."+which+".data.txt", 'r', encoding='utf-8') as f:
        trial_data = f.read()
    trial_data = trial_data.split('\n')
    trial_data = [x.split('\t') for x in trial_data if x != '']
    return trial_data
trial_data = getPairWords("trial")

def pairSimilarity(trial_data, model):
    similarities = np.zeros((len(trial_data,)))
    for k in range(len(trial_data)):
        pair = trial_data[k]
        vector_couple = np.zeros((len(pair), d))
        for i in range(len(pair)):
            word = pair[i]
            vec = np.zeros((d,))
            avg = 0
            for single_word in word.split():
                try:
                    print("Word: ", single_word)
                    # print(type(vec))
                    # print(type(model[single_word.lower()]))
                    if embed != 'elmo':
                        vec = vec + model[single_word.lower()]
                    else:
                        vec = vec + model.embed_sentence([single_word.lower()]).mean(axis=0)
                    avg += 1
                except KeyError:
                    print("New word detected: ", single_word)
                    pass;
            if avg==0:
                avg += 1
            vec = vec/avg
            vector_couple[i,:] = vec
        vec_null = False
        for vector in vector_couple:
            if (vector == np.zeros(vector.shape)).all():
                vec_null = True
        if vec_null:
            sim = 0.5
        else:
            sim = abs(cosine_similarity_single(vector_couple[0], vector_couple[1]))
        similarities[k] = sim*4
        print(pair, sim)
    return similarities
trial_similarities = pairSimilarity(trial_data, model)
print("Similarity: ", trial_similarities)


test_data = getPairWords("test")
test_similarities = pairSimilarity(test_data, model)
print(test_similarities)

outputPathTrial = "SemEval17-Task2/trial/subtask1-monolingual/output/"
outputPathTest = "SemEval17-Task2/test/subtask1-monolingual/output/"
if not os.path.exists(outputPathTrial):
    os.makedirs(outputPathTrial)
if not os.path.exists(outputPathTest):
    os.makedirs(outputPathTest)

def writeOutput(path, similarities):
    with open(path, 'w', encoding='utf-8') as f:
        for x in similarities:
            f.write(str(x) + '\n')
if embed == "elmo":
    embed += which_elmo
trial_path = os.path.join(outputPathTrial, "en.trial."+embed+str(dim)+".txt")
test_path = os.path.join(outputPathTest, "en.test."+embed+str(dim)+".txt")
writeOutput(trial_path, trial_similarities)
writeOutput(test_path, test_similarities)

print("Result saved at %s" % (test_path))


