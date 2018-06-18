from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

class LogReg:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    @classmethod
    def fromCsv(self, fileName):
        reader = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=1)
        data_and_labels = np.matrix(list(reader))
        data = np.matrix(data_and_labels[:, [0, 1, 2]]).astype('float')
        labels = np.array(data_and_labels[:, 3]).astype('intc')

        return LogReg(data, labels)


    def train(self):
        pass

    def plotData(self):
        print('Plotting...')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        blacks = self.labels == 1
        whites = self.labels == 0

        xs = np.extract(blacks, self.data[:, 0])
        ys = np.extract(blacks, self.data[:, 1])
        zs = np.extract(blacks, self.data[:, 2])
        ax.scatter(xs, ys, zs, c='r', marker='o')

        xs = np.extract(whites, self.data[:, 0])
        ys = np.extract(whites, self.data[:, 1])
        zs = np.extract(whites, self.data[:, 2])
        ax.scatter(xs, ys, zs, c='b', marker='^')

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')

        #plt.show()
        fig.savefig('test.png')



