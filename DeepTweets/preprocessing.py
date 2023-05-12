import numpy as np 
import pandas as pd 
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
"""preprocess data by deleting things that are unlikely to help"""

traind=pd.read_csv('training.csv')

def cleaner(t):
    #turning all the text to lowercase 
    t=t.lower() 
    #delete all punctuation
    t = t.translate(str.maketrans('', '', string.punctuation))
    #tokenize
    tokens = nltk.word_tokenize(t)
    c_tokens = []
    for token in tokens:
        #delete all links
        token = re.sub(r'http\S+', '', token)
        if token != '':
            c_tokens.append(token)
    t = ' '.join(c_tokens)
    return t

traind['TweetText']=traind['TweetText'].apply(cleaner)

