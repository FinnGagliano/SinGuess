"""
Neuron inside a layer
"""

class Neuron(object):

    _value = 0

    def __init__(self, value=None):
        """
        Init the neuron
        """
        self.value = value

    @property
    def value(self):
        """
        Value that the neuron is currently storing

        Returns:
            float: the neuron's value
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Sets the value of the neuron

        Args:
            value(float): The value of the neuron, can be None
        """
        self._value = value
