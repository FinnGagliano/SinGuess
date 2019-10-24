"""
Defines network
"""
import numpy as np
from Layer import Layer

class Network(object):
    """
    Class for network, contains all components
    """

    _weightlists = []
    _layers = []

    def __init__(self, layers, weightlists):
        self.layers = layers
        self.weightlists = weightlists

    @property
    def weightlists(self):
        """
        The list of weights between different Layers.

        Returns:
            weightlists(List): List of lists of weight values
        """
        return self._weightlists


    @weightlists.setter
    def weightlists(self, weightlists):
        """
        Sets the weightlists property.

        If a network has 3 layers, then weightlists should have length 2,
        the list of weights between 1st and 2nd layers, and between the 2nd and 3rd layers.

        Args:
            weightlists(List): List of lists of weight values
        """
        if self.layers:
            if len(weightlists) != len(self.layers) - 1:
                raise ValueError(
                    "Should have {} weightlists, gave {}".format(str(len(self.layers) - 1),\
                    len(weightlists))
                )
        self._weightlists = weightlists


    @property
    def layers(self):
        """
        The layers that make up this network

        Returns:
            list: The list of Layer objects
        """
        return self._layers


    @layers.setter
    def layers(self, layers):
        """
        Set layers of Network

        Args:
            layers(List): List of layer objects
        """
        self._layers = layers

    def output(self, input):
        for i in range(len(self.layers) - 1):
            self.layers[i+1].backfeed(self.layers[i], self.weightlists[i][0])

        return self.layers[-1].neurons[0].value


    def get_output_cost_deriv(self, outputs, training_values):
        """
        Finds the cost of the network

        Args:
            output(List): A list of output neurons
            training_values(List): A list of actual values
        Returns:
            cost(List): List of costs
        Raises:
            ValueError: If output length doesn't match training data
        """
        if len(outputs) != len(training_values):
            raise ValueError("Network output length should match training data")

        costs = [outputs[i] - training_values[i] for i in range(len(outputs))]
        for i in range(len(costs)):
             costs[i] *= Layer.deriv_activation(outputs)[i]

        return costs

    def gradient_descent(self, output_cost, learning_rate=1):
        for weightlist in self.weightlists:
            for i in range(len(weightlist)):
                weightlist[i] -= learning_rate * output_cost[i]
