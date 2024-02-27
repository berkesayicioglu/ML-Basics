import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn import linear_model
'''

df=pd.read_csv("homeprices.csv")
print(df)

plt.scatter(df.area,df.price,color='red',marker='*')
plt.show()

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

#print(reg.predict([[3300]]))

d=pd.read_csv("areas.csv")
print(d)
p=reg.predict(d)
d['prices']=p
print(d['prices'])
#exporting value to csv
d.to_csv('areas.csv')
'''

canada_df=pd.read_csv('canada_per_capita_income.csv') #reading CSV
print(canada_df)
reg=linear_model.LinearRegression() #Prediction first step
reg.fit(canada_df[['year']],canada_df.per_capita_income) #Fitting the value 2nd step
print(reg.predict([[2020]])) #importing the year we want as a result
