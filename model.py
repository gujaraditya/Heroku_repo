import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('Iris.csv')
# input
X = dataset.drop(['Id', 'Species'], axis=1)
y = dataset['Species']



xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size = 0.25, random_state = 0)

# sc_x = StandardScaler()
# xtrain = sc_x.fit_transform(xtrain)
# xtest = sc_x.transform(xtest)
# print(xtest[[1]])
# print(xtrain[0:10, :])


LR_classifier = LogisticRegression()
LR_classifier.fit(xtrain, ytrain)

# clf = svm.SVC()
# clf.fit(xtrain, ytrain)

pickle.dump(LR_classifier,open('LR_model.pkl','wb'))

LR_model=pickle.load(open('LR_model.pkl','rb'))
y_pred=LR_model.predict(xtest)
print(LR_model.predict([[6, 3, 4, 2]]))
print(accuracy_score(ytest,y_pred))

