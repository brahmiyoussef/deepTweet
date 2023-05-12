import numpy as np
import pandas as pd
import string
import nltk
import preprocessing as p

#add more rows that contains only words that starts with @ tags 

df=pd.read_csv("training.csv")
arobase = df['TweetText'].str.findall(r'@\w+')
df['TweetText'] = arobase.apply(lambda x: ' '.join(x))

#delete empty rows
empty = ~df['TweetText'].str.contains(r'[a-zA-Z]')
df = df.drop(df[empty].index)
final_df = pd.concat([df, p.traind], axis=0)