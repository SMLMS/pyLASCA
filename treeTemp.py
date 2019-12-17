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
from sklearn import tree
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
predictors = ["area", "lcArea", "contours"]
#predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
print(df.describe())

# =============================================================================
#  creating training and test data set
# =============================================================================
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41)
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

# =============================================================================
# decision tree regressor
# =============================================================================
dtree = tree.DecisionTreeRegressor(max_depth=20, min_samples_leaf=0.05, random_state=41)

dtree.fit(X_train, y_train)

#tree.plot_tree(dtree)

# =============================================================================
# predict on the training data
# =============================================================================
pred_train_tree = dtree.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_tree)))
print(r2_score(y_train, pred_train_tree))

# =============================================================================
# predict on the test data
# =============================================================================
pred_test_tree = dtree.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_tree)))
print(r2_score(y_test, pred_test_tree))

n = 3
print(pred_test_tree[n], y_test[n][0])

# =============================================================================
# plot training data
# =============================================================================
# Plot outputs

plt.scatter(y_test[:,0], X_test[:,2],  color='black')
plt.scatter(pred_test_tree, X_test[:,2] , color='blue')

plt.xticks(())
plt.yticks(())

plt.show()
plt.close()

plt.scatter(y_test[:,0], X_test[:,0],  color='black')
plt.scatter(pred_test_tree, X_test[:,0] , color='blue')

plt.xticks(())
plt.yticks(())

plt.show()
plt.close()

plt.scatter(y_test[:,0], X_test[:,1],  color='black')
plt.scatter(pred_test_tree, X_test[:,1] , color='blue')

plt.xticks(())
plt.yticks(())

plt.show()
plt.close()

tree.plot_tree(dtree)