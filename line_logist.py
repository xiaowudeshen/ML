# -*- coding: utf-8 -*-
import random
from matplotlib import pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
y = []
for i in x:
    temp = 2*i + random.random()
    y.append(temp)

a = 0.001
b = 0.001
for num in range(100000):
    avg_sum = 0
    err = 0
    err_a = 0
    err_b = 0
    for index in range(len(x)):
        err = (a*x[index] + b) - y[index]
        avg_sum += err
        err_a += x[index]*err
        err_b += err**2
    avg_sum = avg_sum/len(x)
    if avg_sum < 0.0001:
        break
    gradient_a = 2*err_a/len(x)
    gradient_b = 2*err_b/len(x)
    a = a - 0.001*gradient_a
    b = b -0.001*gradient_b

print("a=%f\tb=%f"%(a,b))
label = []
for ele in x:
    value = a*ele + b
    label.append(value)
plt.plot(x,y,'r-',x,label,'b')
plt.show()
