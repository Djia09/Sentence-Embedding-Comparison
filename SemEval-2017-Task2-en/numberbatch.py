import io
import numpy as np
import os
import time
path = './../17.06/numberbatch-en-17.06.txt'
with io.open('./../17.06/numberbatch-en-17.06.txt', 'r', encoding='utf-8') as f:
    start = time.time()
    data = f.read()
    print("Loaded GloVe at %s in %fs" % (os.path.basename(path), time.time()-start))

lines = data.split('\n')
print("There are %d lines." % len(lines))

[vocab_size, d] = [int(x) for x in lines[0].split()]
start = time.time()
model = {}
for line in lines[1:]:
    tokens = line.split()
    try:
        model[tokens[0]] = np.array([np.float32(x) for x in tokens[1:]])
    except IndexError as e:
        print("Index Error for: ", tokens)
        print(e)

print("Model GloVe "+os.path.splitext(os.path.basename(path))[0]+" created in %fs" % (time.time()-start))

