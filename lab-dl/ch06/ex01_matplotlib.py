import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d


def fn(x, y):
    # f(x,y) = (1/20) * x**2 + y**2
    return x ** 2 / 20 + y ** 2


def fn_derivative(x, y):
    return x / 10, 2 * y

if __name__ == '__main__' :
    x = np.linspace(-10,10,1000)
    y = np.linspace(-10,10,1000)
    X,Y = np.meshgrid(x,y)
    Z= fn(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    # projection 파라미터를 사용하려면,  mpl_toolkits.mplot3d 패키지가 필요함
    ax.contour3D(X,Y,Z,50)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    # 등고선 그래프
    plt.contour(X,Y,Z,100,cmap='binary')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
