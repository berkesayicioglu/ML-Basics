import pandas as pd
'''
df=pd.read_csv('salaries.csv')

inputs=df.drop('salary_more_than_100k',axis='columns')
target=df['salary_more_than_100k']

from sklearn.preprocessing import LabelEncoder

le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

inputs['company']=le_company.fit_transform(inputs['company'])
inputs['job']=le_company.fit_transform(inputs['job'])
inputs['degree']=le_company.fit_transform(inputs['degree'])

print(inputs)

from sklearn import tree

model=tree.DecisionTreeClassifier()

model.fit(inputs,target) #training

score=model.score(inputs,target)

# print(score)---- score is 1.0

predicted=model.predict([[2,1,0]])

print(predicted)
'''

df=pd.read_csv('titanic.csv')

inputs=df.drop(['Survived','PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
target=df['Survived']

#print(inputs.head())

from sklearn.preprocessing import LabelEncoder

le_pclass=LabelEncoder()
le_sex=LabelEncoder()
le_age=LabelEncoder()
le_fare=LabelEncoder()

inputs['Pclass']=le_pclass.fit_transform(inputs['Pclass'])
inputs['Sex']=le_sex.fit_transform(inputs['Sex'])
inputs['Age']=le_age.fit_transform(inputs['Age'])
inputs['Fare']=le_fare.fit_transform(inputs['Fare'])

print(inputs.head())

from sklearn import tree

model=tree.DecisionTreeClassifier()

model.fit(inputs.values,target.values) #training

score=model.score(inputs.values,target.values)

print("The accuracy rate is {:.2f} ".format(score*100),"%")

pred=model.predict([[0,1,88,42]])

print("[0] means Passed Away\n------------------------->",pred,"\n[1] means Survived")
