import pandas as pd
import numpy as numpy
from sklearn import linear_model
import math
'''
df=pd.read_csv('prices.csv')

bedroom_median=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(bedroom_median)
print(df)

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
pred=reg.predict([[3000,3,40]])
print(pred)

'''

#Exercise

df=pd.read_csv('hiring.csv')

score_median=math.floor(df.test_score.mean())

print(score_median)
df.test_score=df.test_score.fillna(score_median)
print(df,'\n')

reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df.salary)
pred=reg.predict([[2,9,6]])
print(pred)