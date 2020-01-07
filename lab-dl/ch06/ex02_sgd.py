"""
신경망 학습 목적 : 손실함수의 값을 가능한 낮추는 파라미터를 찾는 것
파라미터 (parameter) :
 - 파라미터 : 가중치 (weight), 편향 (bias)
 - 하이퍼파라미터 : 학습률, epoch, batch 크기, 신경망 층에서의 뉴런개수, 신경망 은닉층 개수
ch 06의 목표 : 파라미터를 갱신하는 방법 (SGD, Momentum,Adagrad,Adam)
              하이퍼파라미터를 최적화시키는 방법
"""

class Sgd :
    def __init__(self,learning_rate =0.01):
        self.learning_rate = learning_rate

    def update(self,params,gradients):
        """파라미터 params와 변화율 gradients가 주어지면,
        파라미터들을 갱신하는 메소드드"""
        for key in params :
            params[key] -= self.learning_rate*gradients[key]


if __name__ == '__main__' :
    sgd =