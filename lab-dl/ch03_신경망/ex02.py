

import numpy as np

x = np.array([1,2])
W = np.array([[3,4],[5,6]])
print(x.dot(W))

A= np.arange(1,7).reshape(2,3)
print(A)
B= np.arange(1,7).reshape(3,2)
print(A.dot(B))
print(B.dot(A))

x= np.array([1,2,3])
print(x.shape)

x=x.reshape((3,1))
print(x)
print(x.shape)

