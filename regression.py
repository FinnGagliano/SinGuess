import numpy as np
import matplotlib.pyplot as plt

'''
Basic linear regression for data with added artificial noise. I've found
that these hyperparameters work quite well so far!

training_size = 600
validation_size = 300
testing_size = 100
number_of_parameters = 2
training_iterations = 2000
learning_rate = 0.005

x = np.random.uniform(0, 5, size=(training_size))
xv = np.random.uniform(0, 5, size=(validation_size))
y = (3*x + 2) + np.random.normal(0,0.125,training_size)
'''


training_size = 6000
validation_size = 3000
testing_size = 100
number_of_parameters = 4 # n_o_p=2 for linear regression, n_o_p>2 for non-linear
training_iterations = 1000
learning_rate = 0.00000001

x = np.random.uniform(0, 5, size=(training_size))
xv = np.random.uniform(0, 5, size=(validation_size))
y = np.sin(x)

# Adds some noise to the data
noise = np.random.normal(0,0.125,training_size)
y += noise

# Initializes our weights, cost function and the derivatives of cost
theta = np.random.normal(0,1,number_of_parameters)
J = np.zeros(training_iterations)
DJ = np.zeros((training_iterations,number_of_parameters))

# Iteratively tunes the weights according to derivative of cost of data
for T in range(training_iterations):
    for t in range(training_size):
        h = theta@[x[t]**k for k in range(number_of_parameters)]
        J[T] += ((y[t] - h) ** 2) # Mean Square Error Cost
        for i in range(number_of_parameters):
            DJ[T][i] += 2 * (h-y[t]) * ((x[t]**i) ** i)
    J[T] /= (number_of_parameters * training_size)
    DJ[T] /= (number_of_parameters * training_size)
    theta -= learning_rate * DJ[T]

# Prints final output of weights
print(theta)

h = [0] * validation_size
for V in range(validation_size):
    h[V] = theta@[xv[V]**k for k in range(number_of_parameters)]

plt.scatter(x,y)
plt.scatter(xv,h, color='red')
# Uncomment to see plot of cost function over time
#plt.plot(range(training_iterations), J, color='purple')

plt.show()
