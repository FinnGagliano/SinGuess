import numpy as np
import matplotlib.pyplot as plt

'''
Basic linear regression for data with added artificial noise
'''

training_size = 600
validation_size = 300
testing_size = 100
number_of_parameters = 2 # For scaling up to multivariate regression
training_iterations = 2000
learning_rate = 0.005

x = np.random.uniform(0, 5, size=(training_size))
y = (3*x + 2) + np.random.normal(0,0.125,training_size)
theta = np.random.normal(0,1,number_of_parameters)
J = np.zeros(training_iterations)
DJ = np.zeros((training_iterations,number_of_parameters))

for T in range(training_iterations):
    for t in range(training_size):
        h = theta@[x[t]**k for k in range(number_of_parameters)]
        J[T] += ((y[t] - h) ** 2)
        for i in range(number_of_parameters):
            DJ[T][i] += 2 * (h-y[t]) * ((x[t]**i) ** i)
    J[T] /= (number_of_parameters * training_size)
    DJ[T] /= (number_of_parameters * training_size)
    theta -= learning_rate * DJ[T]

# Prints final output of thetas
print(theta)
xv = np.random.uniform(0, 5, size=(validation_size))
h = [0] * validation_size
for V in range(validation_size):
    h[V] = theta@[xv[V]**k for k in range(number_of_parameters)]

plt.scatter(x,y)
plt.plot(xv,h, color='red')
# Uncomment to see plot of cost function over time
#plt.plot(range(training_iterations), J, color='purple')

plt.show()
