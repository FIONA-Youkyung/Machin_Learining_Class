import numpy as np


def and_gate(x) :
    # x는 [0,0],[0,1],[1,0],[1,1] 중 하나인 numpy.ndarray 타입
    # w =[w1,w2]인 ndarray를 생성
    w = np.array([1,1])
    b= 1
    test = x.dot(w) + b
    if test > 1 :
        return 1
    else :
        return 0

def Nand_gate(x) :
    w = np.array([1, 1])
    b = -1
    test = x.dot(w) + b
    if test <=0 :
        return 1
    else:
        return 0



def Or_gate(x) :
    w = np.array([1, 1])
    b = -1
    test = x.dot(w) + b
    if test <=-1 :
        return 0
    else:
        return 1


if __name__ == '__main__' :
    print(and_gate(np.array([1,1])))

