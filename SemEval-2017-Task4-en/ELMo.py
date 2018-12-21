import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import time

### ELMo embedding on training data
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"

start = time.time()
print("Downloading elmo model...")
elmo = ElmoEmbedder(options_file, weight_file)
print("Downloaded in %fs" % (time.time()-start))

start = time.time()
sentences = ["First sentence .".split(), "Another one".split()]
X = elmo.embed_sentence(sentences[1])
print(X.shape)
print('Type: ', type(elmo))
print("Embedding done in %fs." % (time.time()-start))

