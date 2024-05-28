import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_output(inputs, weights, bias):
    return sigmoid(np.dot(inputs, weights) + bias)

inputs = np.linspace(-10, 10, 100)
true_weights = 2.0
true_bias = 1.0
true_output = calculate_output(inputs, true_weights, true_bias)

weights_range = np.linspace(-3, 3, 5)
for weight in weights_range:
    output = calculate_output(inputs, weight, true_bias)
    plt.plot(inputs, output, label=f'Weight: {weight:.2f}')

plt.plot(inputs, true_output, label='True Output', linestyle='--')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Effect of Varying Weights on Output')
plt.legend()
plt.show()

bias_range = np.linspace(-1, 3, 5)
for bias in bias_range:
    output = calculate_output(inputs, true_weights, bias)
    plt.plot(inputs, output, label=f'Bias: {bias:.2f}')

plt.plot(inputs, true_output, label='True Output', linestyle='--')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Effect of Varying Bias on Output')
plt.legend()
plt.show()
