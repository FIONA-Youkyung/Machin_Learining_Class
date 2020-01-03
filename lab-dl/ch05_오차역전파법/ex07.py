import numpy as np

class Affine :
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.X = None # 입력 행렬을 저장할 field(변수)
        self.doutW = None
        self.doutb = None

    def forward(self,X):
        self.X = X
        out = X.dot(self.W) + self.b
        return out

    def backward(self,dout):
        #db
        self.doutW = X.T.dot(dout)
        doutX = dout.dot(W.T)
        self.doutb = np.sum(dout,axis=0)
        return doutX


if __name__ == "__main__" :
    np.random.seed(103)
    X= np.random.randint(10,size=(2,3))
    W = np.random.randint(10,size=(3,5))
    b = np.random.randint(10,size =5)
    print('b',b)
    affine = Affine(W,b)
    y= affine.forward(X)
    print(y)
    dout=np.random.randn(2,5)
    print('dout',dout)
    dx = affine.backward(dout)
    print('dx',dx)
    print('dw',affine.doutW)
    print('db', affine.doutb)
    # .은 참조 연산자 (reference)
