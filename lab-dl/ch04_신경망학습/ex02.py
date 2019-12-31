import pickle

import numpy as np
from dataset.mnist import load_mnist
from ch03_신경망.ex11 import forward

if __name__ == "__main__" :
    (X_train,y_train),(X_test,y_true) = load_mnist(one_hot_label=True)
    print('y_true =',y_true[:10])

    with open ('../ch03_신경망/sample_weight.pkl','rb') as file :
        network = pickle.load(file)

    y_pred = forward(network,X_test[:10])
    print("y_pred",y_pred)