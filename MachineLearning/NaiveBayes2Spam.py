import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv('spamm.csv')


df.groupby('Category').describe()  #Shows number of spam/ham

df['Spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0 )
print(df.head())

x_train,x_test,y_train,y_test=train_test_split(df.Message,df.Spam,test_size=0.25)
v=CountVectorizer()

x_train_count=v.fit_transform(x_train.values)
model=MultinomialNB()

model.fit(x_train_count,y_train)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))

