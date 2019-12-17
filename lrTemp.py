#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:33:19 2019

@author: malkusch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# =============================================================================
# load data
# =============================================================================

df = pd.read_csv("/home/malkusch/PowerFolders/pharmacology/Daten/ristow/dataframe.csv") 
print(df.shape)
print(df.describe())

# =============================================================================
# scale data
# =============================================================================

target_column = ['thr'] 
predictors = ["area"]#, "lcArea", "contours"]
#predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
print(df.describe())

# =============================================================================
#  creating training and test data set
# =============================================================================
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(X_train.shape)
print(X_test.shape)

# =============================================================================
# linear regressor
# =============================================================================
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)


print("coef", reg.coef_)
print("intercepr", reg.intercept_)

# =============================================================================
# predict on the training data
# =============================================================================
pred_train_lr = reg.predict(X_train)
print("train")
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print(r2_score(y_train, pred_train_lr))
print(reg.score(y_train, pred_train_lr))
u = ((y_train - pred_train_lr) ** 2).sum()
v = ((y_train - pred_train_lr.mean()) ** 2).sum()
print(1-(u/v))
 


# =============================================================================
# predict on the test data
# =============================================================================
pred_test_lr = reg.predict(X_test)
print("test")
print(np.sqrt(mean_squared_error(y_test,pred_test_lr)))
print(r2_score(y_test, pred_test_lr))

n=2
print(pred_test_lr[n], y_test[n][0])

# =============================================================================
# plot training data
# =============================================================================
# Plot outputs
plt.scatter(y_test,X_test,   color='black')
plt.plot(pred_test_lr, X_test, color='blue')

plt.xticks(())
plt.yticks(())

plt.show()

