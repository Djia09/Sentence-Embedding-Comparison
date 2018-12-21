import io
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize

path = "2017_English_final/GOLD/Subtask_A/twitter-2015train-A.txt"

def getTweetsAndLabels(path):
    df = pd.read_csv(path, sep='\t', names=['id', 'sentiment', 'tweet'], encoding='utf-8')
    if path == "2017_English_final/GOLD/Subtask_A/twitter-2016test-A.txt":
        df = df.reset_index()
        df = df[['index', 'id', 'sentiment']]
        df.columns = ['id', 'sentiment', 'tweet']
    tweet_values = list(df['tweet'])
    tweets = preprocess(tweet_values)
    labels = list(df['sentiment'])
    return tweets, labels, list(df['id'])
    
def preprocess(tweet_values):
    regex_url = r'http'
    regex_mention = r'@[a-zA-Z0-9]+'
    regex_hashtag = r'#[a-zA-Z0-9]+'
    regex_consec = '((\D+))\\1\\1'
    regex_time = '([0-9]|[0-2][0-9]):[0-9]{2}'
    emoticons = [":)", ":-)", ":))", ":-))", ":(", ":'(", ":D", "^^", ":/", "/:", ";)", ";p", ":p", ":|"]
    emo_replace = ["happy", "happy", "happy", "happy", "sad", "cry", "laugh", "laugh", "skeptical", "skeptical", "wink","wink>", "cheeky", "neutral"]
    emo_dict = dict(zip(emoticons, emo_replace))
    # print("Emoticons: ", emo_dict)

    tweets = []
    for tweet in tweet_values:
        new_tweet = []
        if type(tweet) == float:
            # print('DEBUG: ', tweet)
            tweet = ''
        for token in tweet.split():
            if re.findall(regex_url, token): #replace urls by "<url>".
                token = token.replace(token, 'url'.upper())
            elif re.findall(regex_mention, token): #replace @... by "<mention>".
                token = token.replace(token, 'mention'.upper())
            elif re.findall(regex_hashtag, token): #replace #... by "<hashtag>".
                token = token.replace(token, 'hashtag'.upper())
            elif token in emo_dict: # replace emoticons.
                token = token.replace(token, emo_dict[token].upper())
            elif re.findall(regex_time, token): # replace time.
                token = token.replace(token, 'time'.upper())
            # elif re.findall(regex_consec, token):
                # print(token, re.findall(regex_consec, token))
            elif re.findall(regex_consec, token):
                # print(token)
                for i in range(len(re.findall(regex_consec, token))):
                    repeat_charac = re.findall(regex_consec, token)[i][0]
                    if len(repeat_charac) > 1 and re.findall('[a-zA-Z0-9]+', repeat_charac) == []:
                        repeat_charac = repeat_charac[0]
                if repeat_charac in list('*$?(){}[]'):
                    repeat_charac = "\\" + repeat_charac
                    # print(repeat_charac)
                # print(repeat_charac)
                while re.sub(repeat_charac*3, repeat_charac*2, token) != token:
                    token = re.sub(repeat_charac*3, repeat_charac*2, token)
                token = re.sub("\W+", " ", token).strip()
            new_tweet.append(token)
        new_tweet = " ".join(new_tweet).lower()
        tweets.append(new_tweet)

    return tweets

