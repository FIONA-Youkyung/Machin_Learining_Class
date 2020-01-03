import numpy as np

np.random.seed(103)
X =np.random.randint(10,size=(2,3))
W= np.random.randint(10,size=(3,5))
print('W=',W)
Z = np.dot(X,W)
print('Z =',Z)

delta = np.random.randn(2,5)
print(delta)

dX = delta.dot(W.T)
print('dX =',dX)
dW = X.T.dot(delta)
print('dW =',dW)


