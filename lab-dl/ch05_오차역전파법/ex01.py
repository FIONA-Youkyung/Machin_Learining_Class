class MultiplyLayer :
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

class AddLayer :
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        return x+y

    def backward(self,delta_output):
        dx = delta_output
        dy = delta_output
        return dx,dy


if __name__ == '__main__' :
    apple_layer = MultiplyLayer()  # MultiplyLayer 객체 생성

    apple = 100  # 사과 한개의 가격: 100원
    n = 2  # 사과 개수: 2개
    apple_price = apple_layer.forward(apple, n)  # 순방향 전파(forward propagation)
    print('사과 2개 가격:', apple_price)

    # tax_layer를 MultiplyLayer 객체로 생성
    # tax = 1.1 설정해서 사과 2개 구매할 때 총 금액을 계산
    tax_layer = MultiplyLayer()
    tax = 1.1
    total_price = tax_layer.forward(apple_price, tax)
    print('토탈 금액:', total_price)

    # f = a*n*t
    # tax가 1 증가하면 전체 가격은 얼마나 증가? df/dt
    # 사과 개수가 1 증가하면 전체 가격은 얼마나 증가? df/dn
    # 사과 가격이 1 증가하면 전체 가격은 얼마나 증가? df/da
    delta = 1.0
    dprice,dtax = tax_layer.backward(delta)
    print(dprice,dtax)
    d_apple_price,d_apple_n = apple_layer.backward(dprice)
    print(d_apple_price,d_apple_n)

    add_layer = AddLayer()
    add_layer_forward = add_layer.forward(100,200)
    print(add_layer_forward)
    add_layer_backward = add_layer.backward(1)
    print(add_layer_backward)
