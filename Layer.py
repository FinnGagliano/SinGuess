"""
Layer class
"""
import numpy as np

class Layer(object):
    """
    Template for Layers of Neural Network
    """

    _size = 0
    _neurons = []

    def __init__(self, neurons):
        """
        Initialize the layer

        Args:
            size(int): number of neurons in layer
            neurons(List): list of neurons in layer
        """
        self.size = len(neurons)
        self.neurons = neurons


    @property
    def neurons(self):
        """
        The set of neurons inside a given layer

        Returns:
            neurons(List): A list of Neuron objects inside this layer
        """

        return self._neurons


    @neurons.setter
    def neurons(self, neurons):
        self._neurons = neurons


    @property
    def size(self):
        """
        The number of neurons in the layer
        """
        return self._size


    @size.setter
    def size(self, size):
        """
        Sets size of Layer

        Args:
            size(int): size of the layer
        """
        self._size = size


    @staticmethod
    def activation(x_in):
        """
        Returns the sigmoid function of the input

        Args:
            x_in(float): The input value of the sigmoid function

        Returns:
            float: The activation function of input

        """
        return np.tanh(x_in)

    @staticmethod
    def deriv_activation(w):
        return 1 - (np.tanh(w) ** 2)


    def backfeed(self, previous_layer, weightlist):
        """
        Sets values of neurons in this layer based on weights with previous layer

        Args:
            previous_layer(Layer): The layer before the current one
            weightlist(List): List of lists of weights in order of neuron
        """
        for i in range(len(self.neurons)):
            self.neurons[i].value = [Layer.activation(weightlist[i] * val) for val in [neuron.value for neuron in previous_layer.neurons][0]]
