import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()

df=pd.DataFrame(iris.data)

df['target']=iris.target

print(df)

x=df.drop(['target'],axis='columns')
y=iris.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
score=model.score(x_train,y_train)

print(score)

'''
digits = load_digits()

df=pd.DataFrame(digits.data)
print(df.head())
df['target']=digits.target

x=df.drop('target',axis='columns')
y=df.target

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=40)
model.fit(x_train,y_train)

print(model.score(x_train,y_train))

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

'''