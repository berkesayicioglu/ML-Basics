import pandas as pd

df=pd.read_csv('carprices.csv')

x=df[['Mileage','Age']]
y=df[['SellPrice']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

clf=LinearRegression()

clf.fit(x_train,y_train)

test=clf.predict(x_test)
print(test)

acc=clf.score(x_test,y_test)
print(acc)






