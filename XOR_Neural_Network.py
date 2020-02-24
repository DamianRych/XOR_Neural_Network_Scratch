import numpy as np
import matplotlib.pyplot as plt
import numpy as np


class Neural_Network():
    def __init__(self, input_data, output_layer, learning_rate):
        self.learning_rate = learning_rate
        self.input_layer = input_data
        
        
        self.weights1 = np.random.rand(self.input_layer.shape[1],2) 
        self.weights2 = np.random.rand(2,1) 
        self.bias1 = 0
        self.bias2 = 0

        self.real_output = output_layer
        self.predicted_output = np.zeros(self.real_output.shape)
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z)) 
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    #feedforward process
    # o----->o
    def prediction(self, input_array):
        self.input_layer = input_array
        #calculation of Layer_2
        self.z1 = np.dot(self.input_layer, self.weights1) + self.bias1
        self.hidden_layer  = self.sigmoid(self.z1)
        
        #calculation of Layer_3(Predicted_Output_layer)
        self.z2 = np.dot(self.hidden_layer, self.weights2)  + self.bias2
        self.predicted_output = self.sigmoid(self.z2)
        return self.predicted_output
    #backpropagation process
    #o<-------o
    def backpropagation(self):
        #calculating the derivatives with the chain rule
        #loss function MSE
        #derivative  of the MSE Function
        error = 2*(self.predicted_output - self.real_output)
        #derivative of the sigmoid function 
        derivative_sigmoid1 = self.sigmoid_derivative(self.predicted_output)
        weights2_update = np.dot(self.hidden_layer.T , error * derivative_sigmoid1)
        
        weights1_update = np.dot(self.input_layer.T,  (np.dot(error * 
                        self.sigmoid_derivative(self.predicted_output), self.weights2.T) * 
                        self.sigmoid_derivative(self.hidden_layer)))
        
        #update the weights with gradient descent
        self.weights2 -= weights2_update * self.learning_rate
        self.weights1 -= weights1_update * self.learning_rate
        
        return error
        
    def training(self, number_of_training):
        loss_value_list = []
        for i in range (0, number_of_training):
            self.prediction(self.input_layer)
            loss_value_list.append(sum(self.backpropagation()))
            
        loss_value_arr = np.array(loss_value_list)
        la = loss_value_arr
        epoch = np.array(range(1,number_of_training + 1))
        epoch = epoch.reshape(number_of_training,1)
        #plot the absolute value of the error
        plt.plot(epoch, abs(loss_value_arr))
        plt.show


def main():
    training_inputs = np.array([[0,0],
                                [0,1],
                                [1,0],
                                [1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    #if the lerning rate is to small it takes longer for the network to learn
    epochs = 10000
    learning_rate = 0.3
    neural_network1 = Neural_Network(training_inputs, training_outputs, learning_rate)
    neural_network1.training(epochs)
    
    #prediction to test how good the network works
    test = np.array([0,1])
    result = neural_network1.prediction(test)
    print(f'Input Array: {test}')
    print(f'Prediction:  {result}')

if __name__ == '__main__':
    main()

