
"""
numpy.nditer 객체 : 반복문을 쓰기 쉽게 도와주는 객체
"""

import numpy as np
np.random.seed(1231)
a = np.random.randint(100 ,size =(2 ,3))
print(a)

for row in a :
    for x in row:
        print(x ,end = ' ')

print()

i = 0
while i <a.shape[0]:
    j = 0
    while j < a.shape[1]:
        print(a[i][j], end=' ')
        j += 1
    i += 1
print()

with np.nditer(a) as iterater :  # nd.nditer 객체를 생성
    for val in iterater :
        print(val, end = ' ')
print()

with np.nditer(a,flags=['multi_index']) as iterater :
    while not iterater.finished :
        i = iterater.multi_index
        print(f'{i} : {a[i]}', end =  ' ')
        iterater.iternext()
print()

with np.nditer(a, flags = ['c_index']) as iterater :
    while not iterater.finished :
        i = iterater.index
        print(f'{i},{iterater[0]}')
        iterater.iternext()
print()

with np.nditer(a,flags = ['c_index'],op_flags =['readwrite']) as it :
    while not it.finished :
        it[0] *=2
        it.iternext()
print(a)