"""
Defines network
"""
import numpy as np

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


    def gradient_descent(self, costs, learning_rate=1):
        for i in range(len(self.weightlists)):
            self.weightlists[i] -= learning_rate * np.gradient(self.weightlists[i])
