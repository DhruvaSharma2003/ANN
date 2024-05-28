import numpy as np
import matplotlib.pyplot as plt

def AND_gate(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    weighted_sum = w1 * x1 + w2 * x2
    return 0 if (weighted_sum <= theta) else 1

def OR_gate(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    weighted_sum = w1 * x1 + w2 * x2
    return 0 if (weighted_sum <= theta) else 1

def NOT_gate(x):
    return 1 if (x == 0) else 0

def XOR_gate(x1, x2):
    s1 = OR_gate(x1, x2)
    s2 = AND_gate(x1, x2)
    return NOT_gate(OR_gate(1 - s1, s2))

x_range = np.linspace(0, 1, 100)
y_range = np.linspace(0, 1, 100)
x_values = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

AND_output = np.array([AND_gate(x[0], x[1]) for x in x_values])
OR_output = np.array([OR_gate(x[0], x[1]) for x in x_values])
XOR_output = np.array([XOR_gate(x[0], x[1]) for x in x_values])

print("AND_output:", AND_output)
print("OR_output:", OR_output)
print("XOR_output:", XOR_output)

def plot_decision_boundary(gate_func, x_range, y_range, label):
    x, y = np.meshgrid(x_range, y_range)
    z = np.zeros_like(x)
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            z[i, j] = gate_func(x[i, j], y[i, j])
    plt.figure(figsize=(15, 5))
    plt.contourf(x, y, z, alpha=0.5, cmap='coolwarm')
    plt.xlabel('Input 1 (x1)')
    plt.ylabel('Input 2 (x2)')
    plt.title(f"{label} Decision Boundary")
    plt.xlim(x_range[0], x_range[-1])
    plt.ylim(y_range[0], y_range[-1])
    plt.colorbar(label='Output')
    plt.show()

# Plot decision boundaries
plot_decision_boundary(AND_gate, x_range, y_range, "AND")
plot_decision_boundary(OR_gate, x_range, y_range, "OR")
plot_decision_boundary(XOR_gate, x_range, y_range, "XOR")
