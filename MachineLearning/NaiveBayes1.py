import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv('titanic.csv')
df.drop(['PassengerId','Name','Ticket','Embarked','Cabin','SibSp','Parch'],axis='columns',inplace=True)


inputs=df.drop('Survived',axis='columns')
target=df.Survived 

dummies=pd.get_dummies(inputs.Sex)
print(dummies.head(4))

inputs=pd.concat([inputs,dummies],axis='columns')
inputs.drop(['Sex','male'],axis='columns',inplace=True)

inputs.Age = inputs.Age.fillna(inputs.Age.mean())


X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)

model=GaussianNB()

model.fit(X_train,y_train)

print("The Accuracy Rate is:{:.2f}".format(model.score(X_test,y_test)*100),"%")

print(model.predict(X_test[0:10]))

