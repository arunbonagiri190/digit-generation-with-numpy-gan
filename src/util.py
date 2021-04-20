import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import time

path = '../data/'

class DataLoader:
    @classmethod
    def load_digit_samples(self):
        return np.load(path+'digit_9_samples.pkl', allow_pickle=True)

    @classmethod
    def generate_noise_samples(self):
        return np.random.random((36, 784))
    
    @classmethod
    def load_data(self):
        return self.load_digit_samples(), self.generate_noise_samples()


class Activations:
    # sigmoid activation function with derivative
    @classmethod
    def sigmoid(self, x, derivative=False):
        if(derivative):
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        
        return 1.0/(1.0 + np.exp(-x))

    # relu activation function with derivative
    @classmethod
    def relu(self, x, derivative=False):
        if(derivative):
            return x > 0
        
        return np.maximum(x, 0)

    # softmax activation function
    @classmethod
    def softmax(self, z):
        exp = np.exp(z)
        return exp / sum(exp)

def save_png(sample, filename):
    plt.imshow(sample.reshape((28, 28)), cmap='Greys')
    plt.title("generated digit")
    plt.savefig('../docs/'+filename+'.png')
    plt.close()

def save_plot(gData, dData, sprt, filename):
    gPlot, = plt.plot(sprt, gData, color='blue', label='g')
    dPlot, = plt.plot(sprt, dData, color='red', label='d')
    plt.legend(handler_map={gPlot: HandlerLine2D(numpoints=4)})
    plt.title("Error plot of Generator(g) and Discriminator(d)")
    plt.savefig('../docs/'+filename+'.png')
    plt.close()