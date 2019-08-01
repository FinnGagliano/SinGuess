from Neuron import Neuron
from Layer import Layer
from Network import Network
import numpy as np
from matplotlib import pyplot

iterations = 40000

# Defines input neuron
input_neuron = [Neuron(180)]
input_layer = Layer(1, input_neuron)

# Defines hidden layer
hidden_neurons = []
for i in range(128):
    hidden_neurons.append(Neuron(round(np.random.ranf(), 2)))
hidden_layer = Layer(128, hidden_neurons)

# Defines output neuron
output_neuron = [Neuron(round(np.random.ranf(), 2))]
output_layer = Layer(1, output_neuron)

# Randomly sets up weights
layers = [input_layer, hidden_layer, output_layer]
weightlists = [[], []]
for weightlist in weightlists:
    for i in range(128):
        weightlist.append(round(np.random.ranf(), 2))

# Initializes network
network = Network(layers, weightlists)

# Trains the network
costs = []
for i in range(iterations):
    x = round(10 * np.random.ranf(), 2)
    y = round(np.sin(x), 2)
    network.layers[0].neurons[0].value = x
    for j in range(len(network.layers) - 1):
        network.layers[j + 1].backfeed(network.layers[j], network.weightlists[j])

    output_value = network.layers[-1].neurons[0].value
    costs.append(network.get_cost(output_value, y))
    network.gradient_descent(costs, 0.1)
    print("Iteration: {}, Cost: {}, Output: {}, Expected Output: {}".format(str(i), str(costs[i]), str(output_value), str(y)))
