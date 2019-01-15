#!/usr/bin/env python

import numpy as np
import os
import time
import io
import math
import argparse
import sys

def parsing_initialisation():
    parser = argparse.ArgumentParser(description="Choose an embedding to compare.")
    parser.add_argument("--embed", help="Available embedding: glove, w2v, miniNumberbatch, elmo")
    parser.add_argument("-d", "--dimension", help="Choose a dimension for GloVe vectors. Default is the smallest: d=50.", type=int)
    parser.add_argument("-p", "--path", help="Path to your embedding model")
    parser.add_argument("--which_elmo", help="Choose ELMo model weights. Default is the smallest: which_elmo=small.")
    args = parser.parse_args()
    embed = args.embed
    if not embed:
        print('No embed argument chosen, argument "embed": %s. Exit program.' % (embed))
        sys.exit()
    elif embed not in ['glove', 'w2v', 'miniNumberbatch', 'elmo']:
        print("Wrong chosen argument 'embed'. Choose between: glove, w2v, miniNumberbatch, elmo")
        sys.exit()
    else:
        print('Chosen embed argument: ', embed)

    print("Begin loading model...")
    if embed == "glove":
        from GloVe import loadAndCreateModel
        if args.path:
            path_to_glove = args.path# "./../../../../Perso/Pretrained-Embedding/GloVe/"
            print('GloVe path: ' + path_to_glove + '. Warning: in GloVe case, it must be the FOLDER path.')
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

    elif embed == "w2v":
        from gensim.models import KeyedVectors
        if args.path:
            path_to_w2v = args.path # './../../../../Perso/Pretrained-Embedding/Word2Vec/GoogleNews-vectors-negative300.bin'
            print('Word2Vec path: ' + path_to_w2v + '. Warning: in W2V case, it must be the FILE.bin path.')
        else:
            print('You need to give Word2Vec FILE.bin path')
            sys.exit()
        model = KeyedVectors.load_word2vec_format(path_to_w2v, binary=True)
        vocab_size = len(model.vocab)
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
        dim = 300
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

def writeOutput(path, similarities):
    with open(path, 'w', encoding='utf-8') as f:
        for x in similarities:
            f.write(str(x) + '\n')

def main():
    parsing_initialisation()
    trial_data = getPairWords("trial")
    trial_similarities = pairSimilarity(trial_data, model)
    print("Similarity: ", trial_similarities)

    test_data = getPairWords("test")
    test_similarities = pairSimilarity(test_data, model)

    outputPathTrial = "SemEval17-Task2/trial/subtask1-monolingual/output/"
    outputPathTest = "SemEval17-Task2/test/subtask1-monolingual/output/"
    if not os.path.exists(outputPathTrial):
        os.makedirs(outputPathTrial)
    if not os.path.exists(outputPathTest):
        os.makedirs(outputPathTest)

    if embed == "elmo":
        embed += which_elmo
    trial_path = os.path.join(outputPathTrial, "en.trial."+embed+str(dim)+".txt")
    test_path = os.path.join(outputPathTest, "en.test."+embed+str(dim)+".txt")
    writeOutput(trial_path, trial_similarities)
    writeOutput(test_path, test_similarities)

    print("Result saved at %s" % (test_path))

if __name__ == '__main__':
    main()