from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import arobase as a
import tunning_hyperparameters as t

testdf=pd.read_csv('test.csv')
X =a.final_df['TweetText']
y =testdf['TweetText']
v = CountVectorizer()
train_vs = v.fit_transform(X)
test_vs = v.transform(y)
mnb = MultinomialNB(alpha=t.best_alpha)
mnb.fit(train_vs,a.final_df['Label'] )
predicted_labels = mnb.predict(test_vs)
result=pd.DataFrame(predicted_labels)
result['TweetId']=testdf['TweetId']
result = result[['TweetId', 0]]
result = result.rename(columns={ 0: 'Label'})
print(result.shape)
print(result)
result.to_csv('final_test1.csv',index=False)
