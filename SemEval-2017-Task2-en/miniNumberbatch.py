import pandas as pd
import time

def loadMiniNumberbatch(path):
    # path = "./../17.06/mini.h5"
    print("Begin load miniNumberbatch at ", path)

    start = time.time()
    df = pd.read_hdf(path)
    print("Loaded mini.h5 in %fs" % (time.time()-start))
    print(df.head())
    print(list(df))
    print(df.shape)
    
    start = time.time()
    df2 = df.reset_index()
    array = df2.values
    print("Type of array: ", type(array))

    en_dict = {}
    for i in range(len(array)):
        fullword = array[i][0]
        if '/c/en' in fullword:
            en_dict[fullword.replace("/c/en/", "")] = array[i][1:]
    print("Model created in %fs" % (time.time()-start))
    return en_dict
# model = loadMiniNumberbatch()
