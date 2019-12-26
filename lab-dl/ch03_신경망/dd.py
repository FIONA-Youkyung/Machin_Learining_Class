def hasard(x) :
    a = x // 10000
    b = x // 1000
    c = x // 100
    d = x // 10
    e = x // 1
    con = a+b+c+d+e
    if x % con == 0 :
        return True
    else :
        return False

print(hasard(9999))