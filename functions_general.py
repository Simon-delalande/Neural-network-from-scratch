import numpy as np
import random 

def relu(x, k):
    if k==0: #0 indicate we are applying the relu function
        return(np.where(x>0, x, np.zeros(x.shape)))

    elif k ==1: #we are applying the derivative
        return(np.where(x>=0, np.ones(x.shape), np.zeros(x.shape)))
    else: pass



def sigmoid(x,k):
    if k==0:
        return(1/(1+ np.exp(-x)) )
    
    elif k==1:
        return np.multiply( sigmoid(x, 0), 1- sigmoid(x, 0))
    else: pass



def cross_entropy(y_hat, y, k):
    if k== 0:
        return np.where(y==1.0, -np.log(y_hat), -np.log(1-y_hat))

    elif k == 1:
        return -y/y_hat + (1-y)/(1-y_hat)
    else: pass



def MSE(y_hat, y, k):
    if k== 0:
        return 1/2*(y_hat-y)**2

    elif k ==1:
        return y_hat-y

    else:
        pass