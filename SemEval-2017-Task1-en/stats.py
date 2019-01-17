import pandas as pd
import io
import re

columns = ["genre", "filename", "year", "ID", "score", "sentence1", "sentence2"]

def csvToDict(path, columns):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    print("There are %d pairs in %s." % (len(data), path))
    table = [dict(zip(columns, x.split('\t'))) for x in data]
    return table
table_train = csvToDict('stsbenchmark/sts-train.csv', columns)
table_dev = csvToDict('stsbenchmark/sts-dev.csv', columns)
table_test = csvToDict('stsbenchmark/sts-test.csv', columns)
table_st = csvToDict('stscompanion/sts-mt.csv', columns[1:])
table_other = csvToDict('stscompanion/sts-other.csv', columns[1:])
table = table_train + table_dev + table_test#  + table_st + table_other
df_train = pd.DataFrame(data=table_train, columns=columns)
df_dev = pd.DataFrame(data=table_dev, columns=columns)
df_test = pd.DataFrame(data=table_test, columns=columns)
df = pd.DataFrame(data=table, columns=columns)
print(df.head())

print("There are %d pairs" % (len(df)))
print("Columns: ", columns)

def process(df):
    df["year"] = df["year"].apply(lambda x: re.sub("2012train", "2012", x))
    df["year"] = df["year"].apply(lambda x: re.sub("2012test", "2012", x))
    df["filename"] = df["filename"].apply(lambda x: re.sub("headlines", "HDL", x))
    df["filename"] = df["filename"].apply(lambda x: re.sub("answers", "Ans.", x))
    df["filename"] = df["filename"].apply(lambda x: re.sub("answer", "Ans.", x))
    df["filename"] = df["filename"].apply(lambda x: re.sub("Question", "Quest.", x))
    df["filename"] = df["filename"].apply(lambda x: re.sub("question", "Quest.", x))
    df["filename"] = df["filename"].apply(lambda x: re.sub("Surprise.", "", x))
    df["filename"] = df["filename"].apply(lambda x: x[0].upper()+x[1:])
    return df
# df_train = process(df_train)
# df_dev = process(df_dev)
# df_test = process(df_test)
df = process(df)
for digit in range(2,8):
    date = "201" + str(digit)
    print("There are %d pairs in %s." % (len(df[df["year"].str.contains(date)]), date))

print(df.dtypes)
# print(df_train.groupby(["filename"]).size())
# print(df_dev.groupby(["filename"]).size())
# print(df_test.groupby(["filename"]).size())
print(df.groupby(["year", "filename"]).size())

with open("./Output_Dev/STS.gs.dev.en-en.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(list(df_dev['score'])))

with open("./Output_Test/STS.gs.test.en-en.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(list(df_test['score'])))
print("Gold scores saved !")
