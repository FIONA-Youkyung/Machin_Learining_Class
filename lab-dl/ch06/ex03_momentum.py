"""
파라미터 최적화 알고리즘
v : 속도 (velocity)
m : 모멘텀 상수 (momentum constant)
Lr : 학습률
W : 파라미터
v = m*v- Lr*dl/dw
W = W +v = W+m*v - Lr*dl/dw
"""
from ch06.ex01_matplotlib import fn_derivative,fn
import matplotlib.pyplot as plt
import numpy as np
from ch06.ex01_matplotlib import fn_derivative


class Monentum :
    def __init__(self,lr,m):
        self.lr = lr
        self.m = m
        self.v = dict() # x방향의 속도와 y빙향의 속도 2가지가 있기 때문


    def update(self,params,gradients):
        if not self.v : # 딕셔너리에 원소가 없으면이라는 뜻
            for key in params :
                self.v[key] = np.zeros_like(params[key])
        for key in params :
            self.v[key] = (self.v[key]*self.m) - (self.lr*gradients[key])
            params[key] += self.v[key]
        # 속도 v, 파라미터 params를 갱신(update)하는 기능을 만들면 됨




if __name__ == '__main__' :
    Mon = Monentum(lr = 0.1,m=0.5)
    init_position = (-7, 2)
    params = dict()
    params['x'], params['y'] = init_position[0], init_position[1]

    gradients = dict()
    gradients['x'], gradients['y'] = 0, 0
    x_history = []
    y_history = []
    for i in range(100):
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'], gradients['y'] = fn_derivative(params['x'], params['y'])
        # fn_derivative는 x,y 각각에 대한 편미분을 나타내는 함수
        Mon.update(params,gradients)
    print(params)
    print(gradients)

    for x, y in zip(x_history, y_history):
        print(f'{x},{y}')

    x = np.linspace(-10,10,2000)
    y = np.linspace(-5,5,1000)
    X,Y = np.meshgrid(x,y)
    Z = fn(X,Y)
    mask = Z>7
    Z[mask] = 0

    plt.contour(X,Y,Z,50)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    # 등고선 그래프에 파라미터(x,y)들이갱신되는 과정을 추가
    plt.plot(x_history,y_history,'o-',color = 'red')
    plt.show()

