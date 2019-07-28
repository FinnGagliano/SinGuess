"""
Layer that produces output value
"""
from Layer import Layer

class OutputLayer(Layer):
    """
    OutputLayer
    """

    def print_my_neurons(self):
        """
        Print the neurons of the OutputLayer
        """
        print(self.neurons)
