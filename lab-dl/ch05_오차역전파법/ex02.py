"""
f(x,y,z) = (x+y)*z
ex_1에서 구현한  multi_add클래스 사용해서 구할 것
"""
from ch05_오차역전파법.ex01 import MultiplyLayer,AddLayer
from ch04_신경망학습.ex05 import numerical_gradient,numerical_diff



x = -2
y = 5
z = -4

layer1 = AddLayer()
layer1_forward = layer1.forward(x,y)
layer2 = MultiplyLayer()
layer2_forward = layer2.forward(layer1_forward,z)
print(layer2_forward)
dt,dz = layer2.backward(1)
print(dt,dz)
dx,dy = layer1.backward(dt)
print(dx,dy)


def f(x,y,z) :
    return (x+y)*z
h = 1e-12
dx = (f(-2+h,5,-4) - f(-2,5,-4)) / h
print(dx)





