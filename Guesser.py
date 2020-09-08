from Neuron import Neuron
from Layer import Layer
from Network import Network
import numpy as np
from matplotlib import pyplot


# Defines input neuron
input_layer = Layer([Neuron()])

# Defines hidden layer
hidden_neurons = []
neuron_number = 100
for i in range(neuron_number):
    hidden_neurons.append(Neuron(round(np.random.ranf(), 2)))
hidden_layer = Layer(hidden_neurons)

# Defines empty output layer
output_layer = Layer([Neuron()])

# Randomly sets up weights
layers = [input_layer, hidden_layer, output_layer]
weightlists = [[] for i in range(len(layers) - 1)]
for i, weightlist in enumerate(weightlists):
    for j in range(len(layers[i].neurons)):
        weightlist.append([round(np.random.ranf(), 2) for k
            in range(len(layers[i + 1].neurons))])

# Initializes network
network = Network(layers, weightlists)

# Trains the network
iterations = 10000
data_size = 1000
layer_costs = []
for i in range(iterations):
    x = np.random.rand(data_size, 1) * 10 - 1
    network.layers[0].neurons = [Neuron([val[0] for val in x])]
    y = [np.sin(x_val) for x_val in x]
    for j in range(len(network.layers) - 1):
        network.layers[j+1].backfeed(network.layers[j], network.weightlists[j][0])

    output_values = network.layers[-1].neurons[0].value
    output_cost = network.get_output_cost_deriv(output_values, y)
    # Keep Learning Rate low - causes weird results after activation
    network.gradient_descent(output_cost, learning_rate=0.1)


pyplot.scatter(x, y)

y = network.output(x)
pyplot.scatter(x, y)

pyplot.show()
