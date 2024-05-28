import numpy as np
import matplotlib.pyplot as plt

# Define your neural network model
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_input_hidden)
        hidden_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_activation, self.weights_hidden_output)
        return output_layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Define the loss function
def loss_function(model, inputs, targets):
    predictions = model.forward(inputs)
    # Example loss function (mean squared error)
    loss = np.mean((predictions - targets) ** 2)
    return loss

# Compute gradient numerically
def gradient_numerical(model, inputs, targets):
    eps = 1e-5
    num_weights_input_hidden = model.weights_input_hidden.size
    num_weights_hidden_output = model.weights_hidden_output.size
    grad_input_hidden = np.zeros_like(model.weights_input_hidden)
    grad_hidden_output = np.zeros_like(model.weights_hidden_output)

    # Compute gradients for input-hidden weights
    for i in range(model.input_size):
        for j in range(model.hidden_size):
            model.weights_input_hidden[i, j] += eps
            loss_plus = loss_function(model, inputs, targets)
            model.weights_input_hidden[i, j] -= 2 * eps
            loss_minus = loss_function(model, inputs, targets)
            grad_input_hidden[i, j] = (loss_plus - loss_minus) / (2 * eps)
            model.weights_input_hidden[i, j] += eps

    # Compute gradients for hidden-output weights
    for i in range(model.hidden_size):
        for j in range(model.output_size):
            model.weights_hidden_output[i, j] += eps
            loss_plus = loss_function(model, inputs, targets)
            model.weights_hidden_output[i, j] -= 2 * eps
            loss_minus = loss_function(model, inputs, targets)
            grad_hidden_output[i, j] = (loss_plus - loss_minus) / (2 * eps)
            model.weights_hidden_output[i, j] += eps

    return grad_input_hidden, grad_hidden_output

# Compute Hessian numerically
def hessian_numerical(model, inputs, targets):
    eps = 1e-5
    num_weights_input_hidden = model.weights_input_hidden.size
    num_weights_hidden_output = model.weights_hidden_output.size
    hess_input_hidden = np.zeros((num_weights_input_hidden, num_weights_input_hidden))
    hess_hidden_output = np.zeros((num_weights_hidden_output, num_weights_hidden_output))

    # Compute Hessian for input-hidden weights
    for i in range(num_weights_input_hidden):
        for j in range(num_weights_input_hidden):
            model.weights_input_hidden.flat[i] += eps
            grad_plus = gradient_numerical(model, inputs, targets)[0]
            model.weights_input_hidden.flat[i] -= 2 * eps
            grad_minus = gradient_numerical(model, inputs, targets)[0]
            hess_input_hidden[i, j] = (grad_plus.flat[j] - grad_minus.flat[j]) / (2 * eps)
            model.weights_input_hidden.flat[i] += eps

    # Compute Hessian for hidden-output weights
    for i in range(num_weights_hidden_output):
        for j in range(num_weights_hidden_output):
            model.weights_hidden_output.flat[i] += eps
            grad_plus = gradient_numerical(model, inputs, targets)[1]
            model.weights_hidden_output.flat[i] -= 2 * eps
            grad_minus = gradient_numerical(model, inputs, targets)[1]
            hess_hidden_output[i, j] = (grad_plus.flat[j] - grad_minus.flat[j]) / (2 * eps)
            model.weights_hidden_output.flat[i] += eps

    return hess_input_hidden, hess_hidden_output

# Newton's method optimization
def newtons_method(model, inputs, targets, max_iter=30, tol=1e-6):
    losses = []
    for i in range(max_iter):
        grad_input_hidden, grad_hidden_output = gradient_numerical(model, inputs, targets)
        hess_input_hidden, hess_hidden_output = hessian_numerical(model, inputs, targets)

        # Flatten gradients and Hessians
        grad_input_hidden_flat = grad_input_hidden.flatten()
        grad_hidden_output_flat = grad_hidden_output.flatten()
        hess_input_hidden_flat = hess_input_hidden.flatten()
        hess_hidden_output_flat = hess_hidden_output.flatten()

        # Concatenate gradients and Hessians
        grad_hess = np.concatenate([grad_input_hidden_flat, grad_hidden_output_flat, hess_input_hidden_flat, hess_hidden_output_flat])

        # Solve the linear system
        update = np.linalg.solve(np.eye(grad_hess.size), grad_hess)

        # Reshape and update weights
        num_weights_input_hidden = model.weights_input_hidden.size
        num_weights_hidden_output = model.weights_hidden_output.size
        update_input_hidden = update[:num_weights_input_hidden].reshape(model.weights_input_hidden.shape)
        update_hidden_output = update[num_weights_input_hidden:num_weights_input_hidden + num_weights_hidden_output].reshape(model.weights_hidden_output.shape)
        model.weights_input_hidden -= update_input_hidden
        model.weights_hidden_output -= update_hidden_output

        # Calculate loss and append to list
        loss = loss_function(model, inputs, targets)
        losses.append(loss)

        if np.linalg.norm(update) < tol:
            break
    return model, losses

# Example usage:
input_size = 2
hidden_size = 3
output_size = 1
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
model = NeuralNetwork(input_size, hidden_size, output_size)
optimized_model, losses = newtons_method(model, inputs, targets)

# Print results
print("Optimized input-hidden weights:")
print(optimized_model.weights_input_hidden)
print("\nOptimized hidden-output weights:")
print(optimized_model.weights_hidden_output)

# Plot loss over iterations
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.show()
