#Figuring out the most important features or principal components
#that has the most impact on target variable

import pandas as pd 
from sklearn.datasets import load_digits

dataset=load_digits()

df= pd.DataFrame(dataset.data,columns=dataset.feature_names)

x=df
y=dataset.target 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=30)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)
score=model.score(x_test,y_test)
#print(score)

from sklearn.decomposition import PCA

pca=PCA(0.95) #use 95% and come up with new dimensions 

x_pca=pca.fit_transform(x)

x_train_pca,x_test_pca,y_train,y_test=train_test_split(x_pca,y,test_size=0.2,random_state=30)

model.fit(x_train_pca,y_train)
pca_score=model.score(x_test_pca,y_test)
print("The score is: {:.2f}".format(pca_score*100),"%")




