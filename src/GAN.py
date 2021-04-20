import numpy as np
from util import Activations as a

class Generator:

    learning_rate = 0.01
    nodes = 784
    np.random.seed(42)

    def __init__(self):
        self.weights = np.array([np.random.normal() for i in range(self.nodes)])
        self.biases = np.array([np.random.normal() for i in range(self.nodes)])

    def __forward__(self, x):
        return a.sigmoid(x * self.weights + self.biases)
    
    def __backward__(self, x, gOut, dOut, discriminator):
        factor = -(1-dOut) * discriminator.weights * gOut * (1-gOut)
        return factor * x, factor   

    def fit(self, x, discriminator):   
        # calculate error
        gOut = self.__forward__(x)
        dOut = discriminator.forward(gOut)
        gError = -np.log(dOut)
        # backpropagate
        nWeights, nBiases = self.__backward__(x, gOut, dOut, discriminator)
        # update
        self.weights -= self.learning_rate * nWeights
        self.biases -= self.learning_rate * nBiases

        return gError

    def generate(self):
        x = np.random.rand()
        return self.__forward__(x)


class Discriminator:

    learning_rate = 0.01
    nodes = 784
    np.random.seed(42)

    def __init__(self):
        self.weights = np.array([np.random.normal() for i in range(self.nodes)])
        self.biases = np.random.normal()

    def forward(self, x):
        return a.sigmoid(np.dot(x, self.weights) + self.biases)
    
    def __backward__(self, x, dOut, isDigit):
        if isDigit:
            nWeights = -x * (1-dOut)
            nBiases = -(1-dOut)
        else:
            nWeights = x * dOut
            nBiases = dOut
        return nWeights, nBiases

    def fit(self, x, isDigit=False):
        # calculate error
        dOut = self.forward(x)
        if isDigit:
            dError = -np.log(dOut)
        else:
            dError = -np.log(1-dOut)
        # back propagation
        nWeights, nBiases = self.__backward__(x, dOut, isDigit)
        # update
        self.weights -= self.learning_rate * nWeights
        self.biases -= self.learning_rate * nBiases

        return dError

