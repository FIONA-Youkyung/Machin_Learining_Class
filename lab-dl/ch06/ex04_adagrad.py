"""
파라미터 최적화 알고리즘 3) Adaptive Gradient
처음에는 큰 학습률로 시작, 점점 학습률을 줄여나가면서 파라미터를 갱신
h = h + grad * grad
Lr = Lr / sqrt(h)
W = W-(Lr/sqrt(h)) *grad
"""

import numpy as np
from ch06.ex01_matplotlib import fn_derivative,fn
import matplotlib.pyplot as plt
from ch06.ex01_matplotlib import fn_derivative


class AdaGrad :
    def __init__(self,lr =0.01):
        self.lr = lr # 학습률
        self.h = dict() # x방향, y 방향 Gradient

    def update(self,params,gradients):
        if not self.h :
            for key in params :
                self.h[key] = np.zeros_like(params[key])
        for key in params :
            self.h[key] += gradients[key] * gradients[key]
            epsilon = 1e-8
            params[key] -= (self.lr/np.sqrt(self.h[key]+epsilon))*gradients[key]

if __name__ == '__main__' :
    Ada = AdaGrad(lr=1.5)

    params = {'x': -7.0, 'y': 2.0}  # 파라미터 초깃값
    print(f"({params['x']}, {params['y']})")
    gradients = {'x': 0.0, 'y': 0.0}  # gradients 초깃값


    x_history = []
    y_history = []
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        # fn_derivative는 x,y 각각에 대한 편미분을 나타내는 함수
        Ada.update(params,gradients)
    print(params)
    print(gradients)

    x = np.linspace(-10,10,2000)
    y = np.linspace(-5,5,1000)
    X,Y = np.meshgrid(x,y)
    Z = fn(X,Y)
    mask = Z>7
    Z[mask] = 0

    plt.contour(X,Y,Z,10)
    plt.xlabel('X')
    plt.ylabel('Y')
    # 등고선 그래프에 파라미터(x,y)들이갱신되는 과정을 추가
    plt.plot(x_history,y_history,'o-',color = 'red')
    plt.show()



