import numpy as np

class neural_net:
    def __init__(self):
        self.layers= {}
        self.size= 0



class layer: 
    def __init__(self, n , activation):
        self.size= n 
        self.activation= activation
        
    def forward_pass(self, X):
        z= self.weights.dot(X)
        a= self.activation(z)
        return(a)
    
    def forward_pass_training(self, X):
        z= self.weights.dot(X) + self.bias
        a= self.activation(z)
        return([z,a])


    def change_weights(self, weights, bias):
        self.weights= weights
        self.bias= bias
    
