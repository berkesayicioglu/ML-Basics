import pandas as pd
import pandas as pd
from matplotlib import pyplot as plt


'''
 df=pd.read_csv('insurance_data.csv')
print(df)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,train_size=0.8)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(x_train,y_train)

pred=model.predict(x_test)

print(pred)

print(model.score(x_test,y_test))

'''

from sklearn.datasets import load_digits
digits = load_digits()

# Get the first digit image

for i in range(5):
 
 plt.imshow(digits.images[i], cmap='gray')
 plt.axis('off')  # Hide axis
 plt.show()


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.2)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=3000)

model.fit(x_train,y_train)

print(model.predict(digits.data[0:5]))

print(model.score(x_test,y_test))
