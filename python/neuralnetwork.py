import numpy as np
import math

class NeuralNetwork:

    # Format is the number of units in each layer, ex: (5, 5, 3) for 3 layer NN
    # with 5 inputs, 5 units in hidden layer, and 3 outputs
    def __init__(self, format, X, y):
        self.format = format
        self.X = X
        self.y = y

        # No. training examples
        self.m = np.size(X, 0)

        # Includes input/output
        self.num_layers = len(format)

        self.Theta = self.genRandWeights()

        print(self.Theta)

    
    @classmethod
    def fromCsv(self, format, fileName):
        """ 
        Construct a neural network using data from a csv file,
        as opposed to passing data directly

        Args:
            format: Network layer architecture. Same as __init__
            fileName: File location of the data and labels, where the
                data is the first n (number of inputs) columns, and 
                the labels is the last column of the csv matrix

        Returns:
            A NeuralNetwork object with the data and labels from the
            csv

        """

        reader = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=1)
        data_and_labels = np.matrix(list(reader))
        data = np.matrix(data_and_labels[:, [0, 1, 2]]).astype('float')
        labels = np.array(data_and_labels[:, 3]).astype('intc')

        return NeuralNetwork(format, data, labels)

    def genRandWeights(self):
        randTheta = []

        # For each matrix of weights
        for i in range(self.num_layers - 1):
            # Range of random values [-ep_init, ep_init]
            ep_init = math.sqrt(6) / math.sqrt(self.format[i] + self.format[i + 1])

            randTheta.append(np.random.rand(self.format[i + 1], self.format[i] + 1))
            randTheta[-1] = randTheta[-1] * 2 * ep_init - ep_init

        return randTheta

    def costFunc(self):
        pass

        
