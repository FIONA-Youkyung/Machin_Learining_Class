"""
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle

from ch03_신경망.ex01 import sigmoid_function
from ch03_신경망.ex05 import softmax
from dataset.mnist import load_mnist
import numpy as np


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어 옴.
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2, W3, b1, b2, b3 shape 확인
    return network

def forward(network,x) :
    """
    순방향 전파 (forward propagation)
    파라미터 x : 이미지 한 개의 정보를 가지고 있는 배열(784,).
    :param network:
    :param x:
    :return:
    """
    # 가중치 행렬 (Weight Matruces
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    #bias matrices
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = x.dot(W1)+b1
    z1 = sigmoid_function(a1)
    # 두 번째 은닉층
    a2 = z1.dot(W2) +b2
    z2 =sigmoid_function(a2)
    #  출력층
    a3 = z2.dot(W3) +b3
    y = softmax(a3)
    return y

def predict(network, X_test):
    """신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
    테스트 데이터의 예측값(배열)을 리턴.
    파라미터  X_test : 10,000개의 테스트 이미지들을 정리 """
    y_pred = [ ]
    for sample in X_test : # 테스트 세트의 각 이미지들에 대해서 반복
        sample_hat = forward(network,sample)# 이미지를 신경망에 전파(통과)시켜서 어떤 숫자가 될지 확률을 계산
        sample_predict =  np.argmax(sample_hat) # 가장 큰 확률의 인덱스
        y_pred.append(sample_predict) # 예측값을 결과 리스트에 추가함
    return y_pred



def accuracy():
    """테스트 데이터 레이블과 테스트 데이터 예측값을 파라미터로 전달받아서,
    정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴."""
    pass

if __name__ == "__main__" :
    (X_train, y_train), (X_test, y_test) = load_mnist(normalize=True,
                                                      flatten=True,
                                                      one_hot_label=False)
    print(X_train[0])
    print(y_train[0])

    # 신경망 가중치(와 편향, bias) 행렬들 생성
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    print(f'W1: {W1.shape}, W2: {W2.shape}, W3: {W3.shape}')
    print(f'b1: {b1.shape}, b2: {b2.shape}, b3: {b3.shape}')

    # 테스트 이미지들의 예측값
    y_pred = predict(network, X_test)
    print('예측값:', y_pred.shape)
    print(y_pred[:10])
    print(y_test[:10])

