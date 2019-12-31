import numpy as np

def numerical_diff(fn,x) :
    """함수 fn과 점 x가 주어졌을 때, x에서의 함수 fn의 미분(도함수) 값"""
    h = 1e-4
    return ((fn(x+h)-fn(x-h))/(2*h))


def f1(x) :
    return (0.001 * x**2) + (0.01 * x)

def f1_prime(x) :
    return (0.002*x) + 0.01

def f2(x) :
    return np.sum(x**2)

def numerical_gradient(fn,x) :
    """fn이 독립변수 여러개를 가지며,  x역시 array를 가짐"""
    x = x.astype(np.float) # 실수 타입으로
    gradient = np.zeros_like(x) #np.zeros(x.shape)
    h = 1e-4
    for i in range(x.size) :
        ith_val = x[i]
        x[i] = ith_val +h
        fh1 = fn(x)
        x[i] = ith_val - h
        fh2 = fn(x)
        gradient[i] = (fh1-fh2) / (2*h)
        x[i] = ith_val
    return gradient

def f3(x) :
    return x[0] +x[1]**2 + x[2]**3


if __name__ == '__main__' :
    print(numerical_diff(f1,3))
    print(f1_prime(3))

    estimate1 = numerical_diff(lambda x : x**2 +4**2,3)
    print(estimate1)
    estimate2 = numerical_diff(lambda  x : 3**2 + x**2,4)
    print(estimate2)
    gradient = numerical_gradient(f2,np.array([3,4]))
    print(gradient)
    gradient2 = numerical_gradient(f3, np.array([1,1,1]))
    print(gradient2)
