"""
Perceptron : 다수의 신호를 입력 받아서, 하나의 신호를 출력

AND :
NAND :
OR :
XOR :

"""

def and_gate(x1,x2) :
    w1,w2 =1,1
    b = -1
    y= (x1*w1) + (x2*w2) +b
    if y >0 :
        return 1
    else:
        return 0

def Nand_gate(x1,x2) :
    w1,w2 =1,1
    b = -1
    y= (x1*w1) + (x2*w2) +b
    if y <=0 :
        return 1
    else:
        return 0
def Or_gate(x1,x2) :
    w1,w2 =1,1
    b = -1
    y= (x1*w1) + (x2*w2) +b
    if y <=-1 :
        return 0
    else:
        return 1

def Xor_gate(x1,x2) :
    # 선형 관계식 하나만 이용해서는 만들 수 없음.
    w1, w2 = 1, 1
    b = -1
    y=and_gate(Nand_gate(x1,x2), Or_gate(x1,x2))

    return y



if __name__ == '__main__' :
    for x1 in (0,1) :
        for x2 in (0,1) :
            print(f'AND {x1},{x2} -> {and_gate(x1,x2)}')

    for x1 in (0,1) :
        for x2 in (0,1) :
            print(f'NAND {x1},{x2} -> {Nand_gate(x1,x2)}')

    for x1 in (0,1) :
        for x2 in (0,1) :
            print(f'Or {x1},{x2} -> {Or_gate(x1,x2)}')

    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f'Xor {x1},{x2} -> {Xor_gate(x1, x2)}')