"""
신경망 학습 목적 : 손실함수의 값을 가능한 낮추는 파라미터를 찾는 것
파라미터 (parameter) :
 - 파라미터 : 가중치 (weight), 편향 (bias)
 - 하이퍼파라미터 : 학습률, epoch, batch 크기, 신경망 층에서의 뉴런개수, 신경망 은닉층 개수
ch 06의 목표 : 파라미터를 갱신하는 방법 (SGD, Momentum,Adagrad,Adam)
              하이퍼파라미터를 최적화시키는 방법
"""
from ch06.ex01_matplotlib import fn_derivative,fn
import matplotlib.pyplot as plt
import numpy as np


class Sgd :
    def __init__(self,learning_rate =0.01):
        self.learning_rate = learning_rate

    def update(self,params,gradients):
        """파라미터 params와 변화율 gradients가 주어지면,
        파라미터들을 갱신하는 메소드드"""
        for key in params :
            params[key] -= self.learning_rate*gradients[key]

if __name__ == '__main__' :
    sgd = Sgd(learning_rate=0.95)

    # ex01. 모듈에서 작성한 fn(x,y) 함수의 최솟값을 임의의 점에서 시작해서 찾아감.
    init_position = (-7,2)
    params = dict()
    params['x'],params['y'] = init_position[0],init_position[1]

    # 각 파라미터에 대한 변화율 (gradient)
    gradients = dict()
    gradients['x'],gradients['y'] = 0,0

    # 각 파라미터들(x,y)을 갱신 값들을 저장할 리스트
    x_history = []
    y_history = []
    for i in range(30) :
        x_history.append(params['x'])
        y_history.append(params['y'])
        gradients['x'],gradients['y'] = fn_derivative(params['x'],params['y'])
        #fn_derivative는 x,y 각각에 대한 편미분을 나타내는 함수
        sgd.update(params,gradients)
    print(params)
    print(gradients)

    for x,y in zip(x_history,y_history) :
        print(f'{x},{y}')


    x = np.linspace(-10,10,200)
    y = np.linspace(-5,5,100)
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



