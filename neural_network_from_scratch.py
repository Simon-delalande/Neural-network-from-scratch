from functions import *
import numpy as np
import matplotlib.pyplot as plt

#modèle simple de classification avec une couche cachée

class Binary_classifier: 

    def __init__(self,n_inputs, n_hidden):

        self.n_inputs= n_inputs
        self.n_hidden= n_hidden

        self.weights1= np.zeros((n_hidden, n_inputs))
        self.weights2= np.zeros((1, n_hidden))
        self.bias1= np.zeros((n_hidden, 1))
        self.bias2= np.zeros(1) 
        self.loss= cross_entropy


    def change_weights(self, W1, W2, b1, b2):
        self.weights1= W1
        self.weights2= W2
        self.bias1= b1
        self.bias2= b2

    def forward_pass(self, input):
        z1= self.weights1.dot(np.reshape(input, (input.shape[0], 1))) + self.bias1
        a1= relu(z1)

        z2= self.weights2.dot(a1) + self.bias2
        a2= sigmoid(z2)
        return(a2)


    def forward_pass_training(self, input):
        z1= self.weights1.dot(np.reshape(input, (input.shape[0], 1))) + self.bias1
        a1= relu(z1)

        z2= self.weights2.dot(a1) + self.bias2
        a2= sigmoid(z2)
        return([z1, a1, z2, a2])


    def back_prop(self, X, y):
        forward= self.forward_pass_training(X)
        z1= forward[0]
        a1= forward[1]
        z2= forward[2]
        y_hat= forward[3]

        delta_out= cross_entropy_derivative(y_hat, y)*sigmoid(z2)*(1-sigmoid(z2))
        delta_in= np.multiply(np.transpose(self.weights2)*delta_out[0], np.reshape(relu_derivative(z1), (self.n_hidden,1)))
        grad_b1= delta_in
        grad_b2= delta_out[0]
        grad_W1= np.reshape(delta_in, (delta_in.shape[0], 1))@ np.reshape(X, (1, X.shape[0]))
        grad_W2= np.reshape(a1*delta_out, (1, a1.shape[0]))

        return [grad_W1, grad_b1, grad_W2, grad_b2]

    def evaluate(self, X, y):
        loss= 0
        for  i in range(X.shape[0]):
            y_hat= model.forward_pass(X[i])
            loss  += self.loss(y_hat,y[i])
        return(loss[0])


    def fit(self, X, y, batch_size, learning_rate, epochs):


        #initialization of random weights 
        W1= np.random.rand(self.n_hidden, self.n_inputs) - 1/2
        W2= np.random.rand(1, self.n_hidden) -1/2
        b1= np.random.rand(self.n_hidden) -1/2
        b1= np.reshape(b1, (self.n_hidden, 1))
        b2= np.random.rand(1) -1/2

        self.change_weights(W1, W2, b1, b2)

        history= []
        for i in range(epochs):
            data_index= np.arange(X.shape[0])

            while data_index.shape[0]> batch_size:
                #select a random sample of size batch size from the data and removing it so all the inputs are considered once during an epoch
                batch_index= np.random.choice(data_index,batch_size , replace= False)
                data_index= np.delete(data_index, batch_index)
                batch_X = np.take(X, batch_index, axis= 0)
                batch_y= np.take(y, batch_index)

                grad_W1, grad_W2, grad_b1, grad_b2= np.zeros((self.n_hidden, self.n_inputs)), np.zeros((1, self.n_hidden)), np.zeros((self.n_hidden, 1)), np.zeros(1)
                for j in range(batch_size):
                    back_prop= self.back_prop(batch_X[j], batch_y[j])
                    grad_W1 += back_prop[0]
                    grad_b1 +=back_prop[1]
                    grad_W2 += back_prop[2]
                    grad_b2 += back_prop[3]
                
                grad_W1, grad_W2, grad_b1, grad_b2=  grad_W1/batch_size, grad_W2/batch_size, grad_b1/batch_size, grad_b2/batch_size
                #actualize weights
                W1, W2, b1, b2= self.weights1- learning_rate*grad_W1, self.weights2 -learning_rate*grad_W2, self.bias1- learning_rate*grad_b1, self.bias2 - learning_rate*grad_b2
                self.change_weights(W1, W2, b1, b2)
                

            if data_index.shape[0]>0:
                batch_index= data_index
                batch_X = np.take(X, batch_index, axis= 0)
                batch_y= np.take(y, batch_index)
                grad_W1, grad_W2, grad_b1, grad_b2= np.zeros((self.n_hidden, self.n_inputs)), np.zeros((1, self.n_hidden)), np.zeros((self.n_hidden, 1)), np.zeros(1)

                for j in range(data_index.shape[0]):
                    back_prop= self.back_prop(batch_X[j], batch_y[j])
                    grad_W1 += back_prop[0]
                    grad_b1 +=back_prop[1]
                    grad_W2 += back_prop[2]
                    grad_b2 += back_prop[3]
                    
                grad_W1, grad_W2, grad_b1, grad_b2=  grad_W1/batch_size, grad_W2/batch_size, grad_b1/batch_size, grad_b2/batch_size

               
                #actualize weights
                W1, W2, b1, b2= self.weights1- learning_rate*grad_W1, self.weights2 -learning_rate*grad_W2, self.bias1- learning_rate*grad_b1, self.bias2 - learning_rate*grad_b2
                self.change_weights(W1, W2, b1, b2)
            history.append(self.evaluate(X,y))
        plt.plot(range(epochs), history)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
            


            


            

model= Binary_classifier(10, 10)
X= (np.random.rand(100, 10)-1/2)*2
y= np.average(X, axis=1)
y= np.where(y>=0, np.ones(100), np.zeros(100))
y= np.reshape(y, (100,1))
model.fit(X ,y, 32, 1e-2, 1000)





            



            



        


