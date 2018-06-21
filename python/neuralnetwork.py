import numpy as np
from scipy.optimize import minimize
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

        self.lambda_const = 1

        # Initiate Theta to random weights
        self.Theta = self.__genRandWeights()

        vec = self.__unrollTheta()

        print(self.costFunc(vec))

    @classmethod
    def fromCsv(self, format, fileName):
        """ 
        Construct a neural network using data from a csv file,
        as opposed to passing data directly. Doesn't work currently,
        only works for color_picker dataset

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
        data_and_labels = np.matrix( list(reader) )

        (m, n) = data_and_labels.shape

        data_range = list(range(0, n - 1))
        data = np.matrix( data_and_labels[:, data_range] ).astype('float')

        labels = np.matrix( data_and_labels[:, n - 1] ).astype('intc')

        labelsMatrix = np.zeros((m, format[-1]), dtype=int)
        for i in range(0, m):
            labelsMatrix[i, labels[i]] = 1

        return NeuralNetwork(format, data, labelsMatrix)

    def train(self):
        unrolled_theta = self.__unrollTheta()
        res = minimize(rosen, unrolled_theta, tol=1e-6)

    def __genRandWeights(self):
        randTheta = []

        # For each matrix of weights
        for i in range(self.num_layers - 1):
            # Range of random values [-ep_init, ep_init]
            ep_init = math.sqrt(6) / math.sqrt(self.format[i] + self.format[i + 1])

            randTheta.append(np.random.rand(self.format[i + 1], self.format[i] + 1))
            randTheta[-1] = np.asmatrix(randTheta[-1] * 2 * ep_init - ep_init)

        return randTheta

    def __unrollTheta(self):
        unrolled_theta = np.array([])

        for mat in self.Theta:
            unrolled_theta = np.append(unrolled_theta, mat.ravel())

        return unrolled_theta

    def __reshapeTheta(self, vec):
        reshaped_theta = []
        start_pos = 0

        for mat in self.Theta:
            end_pos = start_pos + mat.size
            elements = vec[start_pos:end_pos]

            reshaped = np.reshape(elements, mat.shape)
            reshaped_theta.append(reshaped)

            start_pos = end_pos

        return reshaped_theta

    @staticmethod
    def sigmoid(x):
        return 1 / (1+ np.exp(-x))


    def costFunc(self, Theta):
        Theta = self.__reshapeTheta(Theta)
        m = self.m

        J = 0
        
        # Forward Propagation
        a = []
        a.append(np.concatenate((np.ones((m, 1)), self.X), axis=1))
        z = []

        for i in range(self.num_layers - 1):
            next_z = a[i] * Theta[i].T
            z.append(next_z)

            if(i == self.num_layers - 2):
                next_a = self.sigmoid(z[i])
            else:
                ones_array = np.ones((np.size(z[i], 0), 1))
                next_a = np.concatenate((ones_array, self.sigmoid(z[i])), axis=1)

            a.append(next_a)

        # Hypothesis
        hx = a[-1] 

        # Unregularized cost
        J = np.multiply(-self.y, np.log(hx) - np.multiply(1 - self.y, np.log(1 - hx)))
        
        J = (1/m) * J.sum()

        # Regularization term
        reg = 0
        for mat in Theta:
            reg = reg + np.power(mat[:, 1:mat.shape[1]], 2).sum()

        reg = (self.lambda_const / (2 * m)) * reg

        J = J + reg

        return J
            
