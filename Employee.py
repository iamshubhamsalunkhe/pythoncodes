# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:30:28 2019

@author: Shubham
"""
import pandas as pd
from sklearn import linear_model
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_excel("Employee.csv",delimiter='\t')

x = pd.DataFrame(data["Salary"])
y= pd.DataFrame(data["Bonus"])

X_train , x_test ,Y_train , y_test = train_test_split(data,y,test_size = 0.2)

print("X_train data is  :" ,X_train)
print(X_train.shape ,Y_train.shape )
print("y_train data is ",Y_train)
print("Y_test data is ",y_test)
print("X_train data is ",X_train)
print("X_test data is ",x_test)
print(x_test.shape,y_test.shape)
linear = linear_model.LinearRegression()
linear.fit(x,y)
print(linear.score(x,y))
print(linear.intercept_)