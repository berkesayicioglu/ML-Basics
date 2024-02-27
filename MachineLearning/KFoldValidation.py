from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

iris=load_iris()

lr_score=cross_val_score(LogisticRegression(max_iter=1000),iris.data,iris.target)
print(lr_score)

print("Average Logistic Regression Score:{:.2f}".format(np.average(lr_score)*100))

d_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
print(d_scores)

print("Average Decision Tree Score:{:.2f}".format(np.average(d_scores)*100))

s_scores = cross_val_score(SVC(), iris.data, iris.target)
print(s_scores)
print("Average SVM Score:{:.2f}".format(np.average(s_scores)*100))

r_scores = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target)

print(r_scores)
print("Average Random Forest Score: {:.2f}".format(np.average(r_scores)*100))

