# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:48:00 2019

@author: Shubham
"""

from sklearn import linear_model
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

height=[131,164,167,178,152,131,138,189,149,190,175,172,183,155,160,166]
weight=[53,60,62,65,50,45,52,80,49,85,65,70,90,67,54,60]

print("Skewness of height is :",stats.skew(height))
print("kurtosis od height is :",stats.kurtosis(height))
'''skewness of determines left skewness or right skewness '''

print("Skewness of weight is :",stats.skew(weight))
print("kurtosis od weight is :",stats.kurtosis(weight))

print("The correlation between two variables using spearman correlation coefficient is :\n",
      stats.stats.spearmanr(height,weight))
print("The correlation between two variables using pearson correlation coefficient is :\n",
      stats.stats.pearsonr(height,weight))

plt.scatter(weight,height)

plt.xlabel("height")
plt.ylabel("weight")
plt.title("Scatter Plot between height and weight")
plt.show()


x = pd.DataFrame(height)
y = pd.DataFrame(weight)


linear = linear_model.LinearRegression()

linear.fit(x,y)


print("Adjusted R squared using linear regression model is linear regression model : \n",
      linear.score(x,y))

print("Coefficient of independent variables in the  linear regression model is :\n",
      linear.coef_)


print("c- intercept in the linear regression model is :\n",
      linear.intercept_)

newheight = pd.DataFrame([192,160,184])


print("The predicted value for 192 cm height is : \n",
      linear.predict(newheight)[0],"kg.")
print("The predicted value for 160 cm height is : \n",
      linear.predict(newheight)[1],"kg.")

print("The predicted value for 184 cm height is : \n",
      linear.predict(newheight)[2],"kg.")


linear.su







