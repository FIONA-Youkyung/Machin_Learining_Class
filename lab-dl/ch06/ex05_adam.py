"""
4) Adam (Adaptive Moment estimate)

"""
import numpy as np

class Adam :
    def __init__(self,m,lr = 0.001):
        self.lr = lr
        self.m = m
        self.v = dict()
        self.h = dict()
    def update(self,params,gradients):
        b1= 0.9
        b2 = 0.009
        epsilon = 1e-8
        if not self.v : # 딕셔너리에 원소가 없으면이라는 뜻
            for key in params :
                self.v[key] = np.zeros_like(params[key])
        for key in params :
            self.v[key] = (self.v[key]*self.m) - (self.lr*gradients[key])
            params[key] += self.v[key]




