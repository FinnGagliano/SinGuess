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

    def __init__(self, size, neurons):
        """
        Initialize the layer

        Args:
            size(int): number of neurons in layer
            neurons(List): list of neurons in layer
        """
        if size != len(neurons):
            raise ValueError("Number of neurons in layer must match layer size")

        self.size = size
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
    def sigma(x_in):
        """
        Returns the sigmoid function of the input

        Args:
            x_in(float): The input value of the sigmoid function

        Returns:
            float: The sigmoid function of input

        """
        return 1 / (1 + np.exp(-x_in))


    def backfeed(self, previous_layer, weightlist):
        """
        Sets values of neurons in this layer based on weights with previous layer

        Args:
            previous_layer(Layer): The layer before the current one
            weightlist(List): List of weights in order
        """
        for i in range(len(self.neurons)):
            value = 0
            for j, neuron in enumerate(previous_layer.neurons):
                value += weightlist[j] * neuron.value
            self.neurons[i].value = value
