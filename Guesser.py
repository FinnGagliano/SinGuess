from Neuron import Neuron
from Layer import Layer
from Network import Network
import numpy as np

input_neuron = [Neuron(0)]
input_layer = Layer(1, input_neuron)

hidden_neurons = []
for i in range(128):
    hidden_neurons.append(Neuron(np.random.ranf()))
hidden_layer = Layer(128, hidden_neurons)

output_neuron = [Neuron(0)]
output_layer = Layer(1, output_neuron)

layers = [input_layer, hidden_layer, output_layer]
weightlists = [[], []]
for weightlist in weightlists:
    for i in range(128):
        weightlist.append(np.random.ranf())

network = Network(layers, weightlists)

for layer in network.layers:
    print("New Layer: ")
    for neuron in layer.neurons:
        print("Neuron with value = {}".format(neuron.value))

for weightlist in network.weightlists:
    print("weightlist: ")
    for weight in weightlist:
        print("Weight: {}".format(weight))
