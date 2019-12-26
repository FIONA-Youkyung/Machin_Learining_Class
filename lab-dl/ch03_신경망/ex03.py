import numpy as np

x= np.array([1,2])
W= np.array([[1,4],[2,5],[3,6]])
W2 = np.array([[1,2,3],[4,5,6]])


y= W.dot(x)+1
y2= x.dot(W2) +1
print(y)
print(y2)