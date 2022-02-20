import numpy as np
import random 

def relu(x):
    if x>0:
        return(x)
    else: 
        return(0)

def relu_derivative(x):
    if x>0:
        return(1)
    else: 
        return(0)

relu= np.vectorize(relu)
relu_derivative= np.vectorize(relu_derivative)

def sigmoid(x):
    return(1/(1+ np.exp(-x)) )

def cross_entropy(y_hat, y):
    if y == 1.0:
      return -np.log(y_hat)
    else:
      return -np.log(1-y_hat)


def cross_entropy_derivative(y_hat, y):
    return -y/y_hat + (1-y)/(1-y_hat)

sigmoid = np.vectorize(sigmoid)