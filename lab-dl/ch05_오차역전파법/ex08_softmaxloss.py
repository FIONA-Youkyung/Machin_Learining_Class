from ch03_신경망.ex11 import softmax
from ch04_신경망학습.ex03 import cross_entropy
import numpy as np


class SoftmaxwithLoss :
    def __init__(self):
        self.y_true = None  # onehot encoding이 되어 있다고 가정
        self.y_pred = None # softmax 함수의 출력값을 저장하기 위한  field
        self.loss = None

    def forward(self,x,y_true):
        self.y_true = y_true
        self.y_pred= softmax(x)
        self.loss = cross_entropy(self.y_pred,self.y_true)
        return self.loss

    def backward(self,dout =1):
        n = self.y_true.shape[0] # one-hot-encoding의 row 개수
        dx = (self.y_pred-self.y_true) / n # 오차들의 평균
        return dx

if __name__ == "__main__" :
    np.random.seed(103)
    x = np.random.randint(10,size=3)
    print('x =',x)
    y_true = np.array([1.,0.,0.])  # one-hot encoding
    swl = SoftmaxwithLoss()
    loss = swl.forward(x,y_true)
    print(loss)
    y_back = swl.backward()
    print(y_back)
    print(swl.y_pred)