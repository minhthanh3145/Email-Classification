import numpy as np


# Usage, call Constructor, then call method neuralNetwork
class NeuralNetworkClassifier():
    def __init__(self, train_matrix, train_labels, numberOfIterations):
        self.train_matrix = train_matrix
        self.train_labels = train_labels
        self.numberOfIterations = numberOfIterations
        
        
    def trainNeuralNetwork(self):
        print('Training Neural Network ...')
        # we have 3 layers: input layer, hidden layer and output layer
        # input layer has number of nodes equal to number of features
        # hidden layer has 4 nodes
        # output layer has 1 node

        dim1 = len(self.train_matrix[0])
        dim2 = 4
        # randomly initialize the weight vectors
        np.random.seed(1)
        self.weight0 = 2 * np.random.random((dim1, dim2)) - 1
        self.weight1 = 2 * np.random.random((dim2, 1)) - 1
        i = 0
        for j in range(self.numberOfIterations):
            print("iteration", i)
            i = i + 1
            # first evaluate the output for each training email
            layer_0 = self.train_matrix
            layer_1 = sigmoid(np.dot(layer_0, self.weight0))
            layer_2 = sigmoid(np.dot(layer_1, self.weight1))
            # calculate the error
            layer_2_error = self.train_labels - layer_2
            # perform back propagation
            layer_2_delta = layer_2_error * derivativeOfSigmoid(layer_2)
            layer_1_error = layer_2_delta.dot(self.weight1.T)
            layer_1_delta = layer_1_error * derivativeOfSigmoid(layer_1)
            # update the weight vectors
            self.weight1 += layer_1.T.dot(layer_2_delta)
            self.weight0 += layer_0.T.dot(layer_1_delta)
        
        print('Training Neural Network completed')
    

    def predict(self, test_matrix):
        # evaluation on the testing data
        predictResults = []
        layer_0 = test_matrix
        layer_1 = sigmoid(np.dot(layer_0, self.weight0))
        layer_2 = sigmoid(np.dot(layer_1, self.weight1))
        # if the output is > 0.5, then spam else ham
        for i in range(len(layer_2)):
            if(layer_2[i][0] > 0.5):
                predictResults.append(1)
            else:
                predictResults.append(0)



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivativeOfSigmoid(x):
    return x * (1.0 - x)