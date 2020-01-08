"""
파라미터 최적화 알고리즘 6개의 성능 비교 - 손실(loss), 정확도 (accracy)
"""

import matplotlib.pyplot as plt
import numpy as np
from ch05.ex10_twolayer import TwoLayerNetwork
from ch06.ex02_sgd import Sgd
from ch06.ex3_momentum import Momentum
from ch06.ex04_adagrad import AdaGrad
from ch06.ex05_adam import Adam
from ch06.ex06_rmsprop import RMSProp
from ch06.ex07_nesterov import Nesterov
from dataset.mnist import load_mnist

if __name__ == '__main__' :
    #Mnist 데이터 로드
    (X_train,Y_train),(X_test,Y_test) = load_mnist(one_hot_label=True)

    optimizers = dict()
    optimizers['SGD'] = Sgd()
    optimizers['Momentum'] = Momentum()
    optimizers['Adagrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    optimizers['rmsprop'] = RMSProp()
    optimizers['Nesterov'] = Nesterov()
    print(optimizers)

    # 은닉층 1개, 출력층 1개로 이루어진 신경망을 optimizers 개수만큼 생성
    neural_nets = dict()
    train_losses = dict()
    for key in optimizers:
        neural_nets[key] = TwoLayerNetwork(input_size=784,hidden_size=32, output_size=10)
        train_losses[key] =[]

    # 각각의 신경망을 학습시키면서 Loss를 계산, 기록
    iterations = 2_000
    batch_size = 120
    train_size = X_train.shape[0]
    np.random.seed(100)

    for i in range(iterations) :
        batch_mask = np.random.choice(train_size,batch_size)
        # 0~59,999사이의 숫자들 중에서 120개의 숫자를 임의로 선택하겠다.
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]
        # 선택된 학습 데이터 레이블을 사용해서 Gradient 들을 계산
        for key in optimizers :
            gradients = neural_nets[key].gradient(X_batch,Y_batch)
            optimizers[key].update(neural_nets[key].params,gradients)
            loss = neural_nets[key].loss(X_batch,Y_batch)
            train_losses[key].append(loss)

        if i %100 ==0:
            print(f' +++++training #{i}====')
            for key in optimizers:
                print(key,':',train_losses[key][-1])


# 각각의 최적화 알고리즘 별 손실 그래프
    x = np.arange(iterations)
    for key,lossses in train_losses.items() :
        plt.plot(x,lossses,label = key)
        plt.title("Losses")
        plt.xlabel('# of training')
        plt.ylabel('loss')
        plt.legend()
    plt.show()
