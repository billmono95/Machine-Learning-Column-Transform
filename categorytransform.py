

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

df1 = pd.read_csv('movie_review.csv')
X = df1['text']
y = df1['tag']

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)
model = BernoulliNB()
model.fit(X_train,y_train)
p_test = model.predict(X_test)
p_train = model.predict(X_train)
acc_test = accuracy_score(y_test,p_test)
acc_train = accuracy_score(y_train,p_train)

print(f'Accuratezza del test {acc_test}, accuratezza del training {acc_train}')

