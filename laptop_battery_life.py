#!/usr/bin/python

import numpy as np
from numpy.linalg import inv

x = []
y = []

#Input
for i in range(0,100):
    line = raw_input()
    chg_time = float(line.split(",")[0])
    lst_time = float(line.split(",")[1])
    if chg_time<4:
        x.append([1.0,chg_time])
        y.append(lst_time)


#Train
x = np.matrix(np.array(x,dtype=float))
y = np.matrix(np.array(y,dtype=float))
y = np.transpose(y)
theta = inv(np.transpose(x)*x)*np.transpose(x)*y

#Predict
theta = [[-1.1379786e-15], [2.0]]
time_charged = float(raw_input().strip())
if time_charged < 4.0:
    x = np.matrix(np.array([1.0,time_charged],dtype=float))
    time_lasted = x * theta
    print(time_lasted.item(0))
else:
    print(8.0)


