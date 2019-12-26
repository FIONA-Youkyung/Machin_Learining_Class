"""
Perceptron :
- 입력 [x1,x2]
- 출력 y= x1*w1 +x2*w2 + b

"""
import numpy as np
import math
import matplotlib.pyplot as plt


def step_function(x) :
    y = x > 0
    return y.astype(np.int)

def sigmoid_function(x) :
    #print (1/(1+math.exp(-x)))
    return (1/(1+np.exp(-x)))

def relu(x) :
    """
    rectified linear Unit
    :param x:
    :return:
    """
    y = x >0
    return np.maximum(0,x)




if __name__ == "__main__" :
    x= np.arange(-5,5,0.01)
    print(relu(x))
    y= relu(x)
    plt.plot(x,y)
    plt.show()


