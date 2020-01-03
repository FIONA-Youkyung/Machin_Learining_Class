class multiplyLayer :
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        return x*y

    def tax_layer(self,x,t):
        return x*t

    def backward(self,delta_output):
        dx = delta_output * self.y
        dy = delta_output * self.x
        return dx,dy

class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, delta_output):
        dx = delta_output
        dy = delta_output
        return dx, dy

if __name__ == '__main__' :
    Apple_n = 2
    Apple_price = 100
    Orange_n = 3
    Orange_price = 150
    t=1.1

    Layer_1= multiplyLayer()
    Layer1 = Layer_1.forward(Apple_n,Apple_price)
    Layer1_1 = multiplyLayer()
    Layer11 =  Layer1_1.forward(Orange_n,Orange_price)
    Layer_2 = AddLayer()
    Layer2=Layer_2.forward(Layer1,Layer11)
    Layer_3 = multiplyLayer()
    Layer3=Layer_3.forward(Layer2,t)
    print(Layer3)
    L3x,L3y = Layer_3.backward(1)
    print(L3x,L3y)
    L2x,L2y = Layer_2.backward(L3x)
    print(L2x,L2y)
    L1x,L1y = Layer_1.backward(L2x)
    print(L1x,L1y)
    L1_1x,L1_1y = Layer1_1.backward(L2y)
    print(L1_1x,L1_1y)

