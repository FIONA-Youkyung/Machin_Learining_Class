"""sigmoid
dy/dx = y(1-y) 증명
sigmoid 뉴런 (forward, backward)
"""
import numpy as np

class Sigmoid :
    def __init__(self):
        pass

    def forward(self,x):
        y=1/(1+np.exp(-x))
        self.y = y
        return y

    def backward(self,dout):
        return dout*self.y*(1-self.y)


if __name__ == "__main__" :
    sigmoid_gate = Sigmoid()
    y= sigmoid_gate.forward(x=0.)
    print(y)
    # x= 0 에서의 sigmoid의 gradient(접선의 기울기)

    de = sigmoid_gate.backward(dout=1.)
    print(de)

    h= 1e-7
    dx2 = (sigmoid(0.+h) - sigmoid(0.)) / h
    print(dx2)
