import numpy as np
from functions_general import *
import matplotlib.pyplot as plt


class layer: 
#attributes: weights -> matrix , bias -> vector , activation -> function 

    def __init__(self, n , activation):
        self.size= n 
        self.activation= activation
        
    def forward_pass(self, X):
        z= self.weights.dot(X)
        z += self.bias 
        a= self.activation(z, 0)
        return(a)
    
    def forward_pass_training(self, X):
        z= self.weights.dot(X)
        z += self.bias
        a= self.activation(z, 0)
        return([z,a])


    def change_weights(self, weights, bias):
        self.weights= weights
        self.bias= bias
    

class neural_net:
    def __init__(self):
        self.layers= {}
        self.size= 0

    def add_input_layer(self, n):
        self.input_shape= n 

    def add_layer(self, n, activation):
        self.size +=1
        self.layers[self.size]= layer(n, activation)


    def forward_pass(self, X):
        for i in range(1,self.size+1):
            X= self.layers[i].forward_pass(X)
        return(X)

    def forward_pass_training(self, X):
        A= {0: X} #history of the activations of the layers
        Z= {} #history of the outputs before the activation function of the layers
        for i in range(1,self.size+1):
            result= self.layers[i].forward_pass_training(X)
            z= result[0]
            X= result[1]
            A[i]= X
            Z[i]= z
        

        return [A, Z]


    def initialize_weights(self):
        if self.size >0:
            W= (np.random.rand(self.layers[1].size, self.input_shape) - 1/2)*2
            B= (np.random.rand(self.layers[1].size) -1/2)*2
            self.layers[1].change_weights(W, B)

        if self.size >1:
            for i in range(2, self.size+1):
                W= (np.random.rand(self.layers[i].size, self.layers[i-1].size) - 1/2)*2
                B= (np.random.rand(self.layers[i].size) -1/2)*2
                self.layers[i].change_weights(W, B)

        else: 
            pass

    def compile(self, loss):
        self.loss= loss


    def back_prop(self, X, y):
        loss= self.loss
        grad_weights= {}
        grad_bias = {}
        forward= self.forward_pass_training(X)

        #error last layer
        y_hat= forward[0][self.size]
        z= forward[1][self.size]
        gradient_loss= loss(y_hat, y, 1)
        activation_derivative= self.layers[self.size].activation(z, 1) #1 for the derivative 
        delta= np.multiply(gradient_loss,activation_derivative)
        grad_bias[self.size]= delta

        grad_W= np.reshape(delta, (delta.shape[0], 1)).dot(np.reshape(forward[0][self.size-1],(1, forward[0][self.size-1].shape[0]) ))
        grad_weights[self.size]= grad_W

        for i in range(self.size-1, 0, -1):
            delta= np.transpose(self.layers[i+1].weights).dot(delta)
            sigma= self.layers[i].activation(forward[1][i],1)
            delta= np.multiply(delta, sigma )
            grad_bias[i]= delta

            grad_W= np.reshape(delta, (delta.shape[0], 1)).dot(np.reshape(forward[0][i-1],(1, forward[0][i-1].shape[0]) ))
            grad_weights[i]= grad_W

    
        return [grad_weights, grad_bias]

    def evaluate(self, X, y):
        l=0
        for i in range(X.shape[0]):
            y_hat=  self.forward_pass(X[i])
            l+= self.loss(y_hat, y[i], 0)
        return(l)


    def fit(self, X, y, X_val, y_val, batch_size, learning_rate, epochs):

        self.initialize_weights()
        history= []
        history_val= []

        for i in range(epochs):

            data_index= np.arange(X.shape[0])

            while data_index.shape[0]> batch_size:

                batch_index= np.random.choice(data_index,batch_size , replace= False)
                data_index= np.delete(data_index, batch_index)
                batch_X = np.take(X, batch_index, axis= 0)
                batch_y= np.take(y, batch_index, axis= 0)

                backprop= self.back_prop(batch_X[0], batch_y[0])
                grad_w= backprop[0]
                grad_b= backprop[1]

                if batch_size>1:

                    for j in range(1, batch_size):
                        back= self.back_prop(batch_X[j], batch_y[j])
                        for l in range(1, self.size+1):
                            grad_w[l]+= back[0][l]
                            grad_b[l]+= back[1][l]

                for l in range(1,self.size+1):
                    W= self.layers[l].weights
                    B= self.layers[l].bias
                    W= W- learning_rate*grad_w[l]/batch_size
                    B= B- learning_rate*grad_b[l]/batch_size

                    self.layers[l].change_weights(W, B) 

            if data_index.shape[0]>0:
                batch_index= data_index
                batch_X = np.take(X, batch_index, axis= 0)
                batch_y= np.take(y, batch_index, axis= 0)

                backprop= self.back_prop(batch_X[0], batch_y[0])
                grad_w= backprop[0]
                grad_b= backprop[1]

                if batch_size>1:

                    for j in range(1, data_index.shape[0]):
                        back= self.back_prop(batch_X[j], batch_y[j])
                        for l in range(1, self.size+1):
                            grad_w[l]+= back[0][l]
                            grad_b[l]+= back[1][l]

                for l in range(1, self.size+1):
                    W= self.layers[l].weights
                    B= self.layers[l].bias
                    W= W- learning_rate*grad_w[l]/batch_size
                    B= B- learning_rate*grad_b[l]/batch_size

                    self.layers[l].change_weights(W, B) 

            lss= self.evaluate(X,y)
            lss_val= self.evaluate(X_val, y_val)
            history.append(lss[0])
            history_val.append(lss_val[0])

        plt.plot(range(epochs), history, 'r', label= 'loss')
        plt.plot(range(epochs), history_val, 'b', label= 'validation loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()






model= neural_net()
model.add_input_layer(10)
model.add_layer(16, relu)
model.add_layer(32, relu)
model.add_layer(1, sigmoid)
model.compile(cross_entropy)

X= np.random.rand(500, 10)-1/2
y= np.average(X, axis=1)
y= np.where(y>=0, np.ones(500), np.zeros(500))
y= np.reshape(y, (500,1))

X_val= np.random.rand(200, 10)-1/2
y_val=  np.average(X_val, axis=1)
y_val= np.where(y_val>=0, np.ones(200), np.zeros(200))
y_val= np.reshape(y_val, (200,1))

model.fit(X,y,X_val, y_val, 32, 1e-2, 700)


'''
y= np.average(X, axis=1)

X_val= np.random.rand(400, 10)-1/2
y_val= np.average(X_val, axis=1)

model.fit(X,y,X_val, y_val, 16, 1e-3, 100)
'''