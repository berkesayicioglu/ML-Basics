#hata var -- will be fixed

import pandas as pd 
from sklearn.linear_model import LinearRegression
model=LinearRegression()
'''
merged=pd.concat([df,int_dummies],axis='columns')
#print(merged)

final=merged.drop(['town','west windsor'],axis='columns')
print(final)



x=final.drop(['price'],axis='columns')

y=final.price

model.fit(x,y) #to train your model

pred=model.predict([[2800,0,1]]) #to predict the given values with trained model
#print(pred)

#model.score(x,y) ---- model accuracyi bulmak icin bu kullaniliyor
'''
df=pd.read_csv('housingprices.csv')

dummies=pd.get_dummies(df.town)
int_dummies=dummies.astype(int)



from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

dfle=df
dfle.town=le.fit_transform(dfle.town)
print(dfle)

x_val=dfle[['town','area']]
y_val=dfle[['price']]

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()
from sklearn.compose import ColumnTransformer
import numpy as np
ct = ColumnTransformer([("town", OneHotEncoder(), [0])], remainder = 'passthrough')


x_val = np.array(ct.fit_transform(x_val), dtype=float) 

y_val=np.array(ct.fit_transform(y_val),dtype=float)

model.fit(x_val,y_val)

predicted=model.predict([[1,0,2800]])
print(predicted)



