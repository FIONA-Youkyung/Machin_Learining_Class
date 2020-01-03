import numpy as np

class Relu :
    """
    Rectified Liniar Unit
    Relu prime :
    """
    def __init__(self):
        # relu 함수의 input 값이 0q보다 큰지 작은지 저장할 field
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
        return np.maximum(0,x)

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx
if __name__ == "__main__":
    relu_gate = Relu()
    np.random.seed(103)
    x= np.random.randn(5)
    a = relu_gate.forward(x)

    print(x)
    print(relu_gate.mask)

