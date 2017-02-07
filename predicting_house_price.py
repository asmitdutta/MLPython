#!/usr/bin/python

import numpy as np
from numpy.linalg import inv

line = raw_input()
arr = line.split(" ")
n = int(arr[0])
m = int(arr[1])
x = []
y = []

#Input
for i in range(0,m):
   line = raw_input()
   arr = line.split(" ")
   xm = np.concatenate(([1],arr[0:n]),axis=0)
   x.append(xm)
   y.append(arr[n])

#Calculate theta 
x = np.matrix(np.array(x,dtype=float))
y = np.matrix(np.array(y,dtype=float))
y = np.transpose(y)
theta = inv(np.transpose(x)*x)*np.transpose(x)*y

#Input
line = raw_input()
arr = line.split(" ")
m = int(arr[0])
x = []

for i in range(0,m):
   line = raw_input()
   arr = line.split(" ")
   xm = np.concatenate(([1],arr[0:n]),axis=0)
   x.append(xm)
   
#Predict house price  
x = np.matrix(np.array(x,dtype=float))
price = x*theta 

for i in range(len(price)):
   print(price.item(i))