"""
신경망 층들을 지나갈 때, 사용되는 가중치, 편향 행렬들을 찾는 게 목적

"""
from dataset.mnist import load_mnist
import numpy as np

if __name__ == "__main__" :
    (X_train,y_train),(X_test,y_true) = load_mnist()
    print('y_true =',y_true[:10])

    # 10 테스트 이미지들의 실제 값
    y_pred = np.array([7,2,1,6,4,1,4,9,6,9])
    print('y_pred =', y_pred[:10])

    # 오차
    error = y_pred -y_true[:10]
    print(error)

    sq_err = error **2
    print(f'squared error = {sq_err}')

    # 평균 제곱 오차
    mse = np.mean(sq_err)
    print(f'squared error ={mse}')
    print(f'RMSE ={np.sqrt(mse)}')
