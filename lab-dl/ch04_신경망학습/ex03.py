"""
교차 엔트로피 (Cross entropy error)
"""
import pickle
from ch03_신경망.ex11 import forward
from dataset.mnist import load_mnist
import numpy as np
import math

def _cross_entropy(y_pred,y_true) :
    delta = 1e-7
    return -np.sum(y_true * np.log(y_pred + delta))

def cross_entropy(y_pred,y_true) :
    if y_pred.ndim == 1:
        ce = _cross_entropy(y_pred, y_true)
    elif y_pred.ndim == 2:
        ce = _cross_entropy(y_pred, y_true) / len(y_pred)
    return ce




if __name__ == "__main__" :
    (X_train,y_train),(X_test,y_true) = load_mnist(one_hot_label=True)

    print('y_true =',y_true[:10])

    with open('../ch03_신경망/sample_weight.pkl', 'rb') as file:
        network = pickle.load(file)
    y_pred = forward(network, X_test[:10])

    print('y_true[0] =',y_true[0]) # 숫자 7 이미지
    print('y_pred[0] =',y_pred[0]) #
    np.random.seed(1227)
    y_true = np.random.randint(10, size=10)
    print(y_true)
    y_pred = np.array([4,3,9,7,3,1,6,6,8,8])
    y_true2 = np.zeros((y_true.size,10))
    for i in range(y_true.size) :
        y_true2[i][y_true[i]] = 1
    print(y_true2)
    print('cross_entropy',cross_entropy(y_pred,y_true2))



