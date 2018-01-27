# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
import random
from matplotlib import pyplot as plt
# import numpy as np
mode = LinearRegression()
x = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19]]
y = []
x_value = []
for i in x:
    x_value.append(i[0])
    temp = 2*i[0] + random.random()
    y.append(temp)
mode.fit(x,y)
print(mode.coef_)
print(mode.intercept_)
label = mode.predict(x)
plt.plot(x_value,y,'r-',x_value,label,'b')
plt.show()
