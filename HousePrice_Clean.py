# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:08:00 2020

@author: ugur
"""


import pandas as pd
from sklearn.model_selection import train_test_split

#Read data
X_full = pd.read_csv("train.csv",index_col ='Id')
print(X_full['SalePrice'].describe())


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Correlation Matrix
corrmat = X_full.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

k = 10 # number of the variable for the heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(X_full[cols].values.T)
sns.set(font_scale = 1.25)

hm = sns.heatmap(cm,cbar = True, annot = True, square = True, fmt = '.2f', 
                 annot_kws = {'size': 10}, yticklabels = cols.values, 
                 xticklabels = cols.values)
plt.show()

features = ['OverallQual','GrLivArea','GarageArea','TotalBsmtSF','YearBuilt']

#Remove rows with missing target
X_full.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis = 1, inplace = True)

X = X_full[features]

#lets check the missing data in our chosen features

total = X.isnull().sum().sort_values(ascending = False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent], axis = 1, keys=['Total','percent'])
print(missing_data.head(20))

X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size = 0.7,
                                                      test_size = 0.3, random_state = 0)

from xgboost import XGBRegressor

#Evaluation Metric
from math import sqrt

#XGBRegressor
model_2 = XGBRegressor(n_estimators = 500, learning_rate = 0.05)
model_2.fit(X_train,y_train,early_stopping_rounds = 5,
            eval_set = [(X_valid,y_valid)], verbose = False)

from sklearn.model_selection import cross_val_score
score = -1 * cross_val_score(model_2,X,y,cv = 5, scoring = 'neg_mean_absolute_error')
rsme_crossVal = sqrt(score.mean())
print("RSME (XGBRegressor) by using CrossValidation:", rsme_crossVal)



