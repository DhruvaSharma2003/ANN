import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_vals = np.exp(x - np.max(x))
    return exp_vals / np.sum(exp_vals)

def linear(x):
    return x

def hard_limiter(x, threshold=0.5):
    return np.where(x > threshold, 1, 0)

X = np.array([[0.1, 0.2],
              [0.2, 0.3],
              [0.4, 0.5]])
W = np.array([[0.5, 0.3],
              [0.2, 0.4]])
b = np.array([0.1, 0.2])

def forward(X, W, b, activation_func):
    return activation_func(np.dot(X, W) + b)

print("Sigmoid Activation:")
print(forward(X, W, b, sigmoid))
print("\nReLU Activation:")
print(forward(X, W, b, relu))
print("\nTanh Activation:")
print(forward(X, W, b, tanh))
print("\nSoftmax Activation:")
print(forward(X, W, b, softmax))
print("\nLinear Activation:")
print(forward(X, W, b, linear))
print("\nHard Limiter Activation:")
print(forward(X, W, b, hard_limiter))

x = np.linspace(-5, 5, 100)
sigmoid_y = sigmoid(x)
relu_y = relu(x)
tanh_y = tanh(x)
softmax_y = softmax(x)
linear_y = linear(x)
hard_limiter_y = hard_limiter(x)

plt.figure(figsize=(12, 10))
plt.plot(x, sigmoid_y, label='Sigmoid', color='blue')
plt.plot(x, relu_y, label='ReLU', color='red')
plt.plot(x, tanh_y, label='Tanh', color='green')
plt.plot(x, softmax_y, label='Softmax', color='purple')
plt.plot(x, hard_limiter_y, label="Hard Limiter(threshold=0.5)", color='pink')
plt.plot(x, linear_y, label='Linear', color='orange', linestyle='--')
plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
