import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import arobase as a

traindf =a.final_df
param = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
mnb = MultinomialNB()
grid = GridSearchCV(mnb, param, cv=5)
v = CountVectorizer()
train_vs = v.fit_transform(traindf['TweetText'])
print('data vectorized')
mnb.fit(train_vs, traindf['Label'])
grid.fit(train_vs, traindf['Label'])
best_alpha = grid.best_params_['alpha']
print('optimal alpha:', best_alpha)

