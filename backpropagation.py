#without momentum
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

w11, w21 = 0.5, 0.11
w12, w22 = 0.2, 0.11
w32, w42 = 0.8, 0.9
b1, b2, b3, b4 = 0.1, 0.2, 0.3, 0
w31, w41 = 0.7, 0.6

x1, x2 = 0.8, 0.6
h1 = w11 * x1 + w21 * x2 + b1
h2 = w12 * x1 + w22 * x2 + b2

o5 = sigmoid(w31 * h1 + w41 * h2 + b3)
o6 = sigmoid(w32 * h1 + w42 * h2 + b4)

y5, y6 = 0.9, 0.7

E = 0.5 * ((o5 - y5)**2 + (o6 - y6)**2)

print(f"Output o5 (sigmoid): {o5:.4f}, Output o6 (sigmoid): {o6:.4f}")
print(f"Mean Squared Error (E): {E:.4f}")

mse_values = []
epoch_values = []

for epoch in range(10):
    h1 = w11 * x1 + w21 * x2 + b1
    h2 = w12 * x1 + w22 * x2 + b2

    o5 = sigmoid(w31 * h1 + w41 * h2 + b3)
    o6 = sigmoid(w32 * h1 + w42 * h2 + b4)

    delta5 = (o5 - y5) * o5 * (1 - o5)
    delta6 = (o6 - y6) * o6 * (1 - o6)

    delta1 = ((delta5 * w31) + (delta6 * w32)) * h1 * (1 - h1)
    delta2 = ((delta5 * w41) + (delta6 * w42)) * h2 * (1 - h2)

    learning_rate = 0.1

    w11 += learning_rate * delta1 * x1
    w21 += learning_rate * delta1 * x2
    w12 += learning_rate * delta2 * x1
    w22 += learning_rate * delta2 * x2
    w31 += learning_rate * delta5 * o5
    w41 += learning_rate * delta6 * o6
    w32 += learning_rate * delta5 * o5
    w42 += learning_rate * delta6 * o6

    b1 += learning_rate * delta1
    b2 += learning_rate * delta2
    b3 += learning_rate * delta5
    b4 += learning_rate * delta6

    E = 0.5 * ((o5 - y5) + (o6 - y6))
    mse_values.append(E)
    epoch_values.append(epoch + 1)

    print(f"Epoch {epoch + 1}: Mean Squared Error = {E:.4f}")

plt.plot(epoch_values, mse_values, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs')
plt.grid(True)
plt.show()

#with momentum
import numpy as np
import matplotlib.pyplot as plt

w11, w21 = 0.5, 0.3
w12, w22 = 0.2, 0.4
w31, w41 = 0.7, 0.6
w32, w42 = 0.8, 0.9
b1, b2, b3, b4 = 0.1, 0.2, 0.3, 0.4
x1, x2 = 0.8, 0.6

h1 = w11 * x1 + w21 * x2 + b1
h2 = w12 * x1 + w22 * x2 + b2

o5 = sigmoid(w31 * h1 + w41 * h2 + b3)
o6 = sigmoid(w32 * h1 + w42 * h2 + b4)

y5, y6 = 0.9, 0.7

E = 0.5 * ((o5 - y5)**2 + (o6 - y6)**2)
print(f"Output o5 (sigmoid): {o5:.4f}, Output o6 (sigmoid): {o6:.4f}")
print(f"Mean Squared Error (E): {E:.4f}")

w11, w21, w12, w22, w31, w41, w32, w42 = np.random.rand(8)
b1, b2, b3, b4 = np.random.rand(4)
x1, x2 = 0.5, 0.3
y5, y6 = 0.9, 0.7

mse_values = []
epoch_values = []

for epoch in range(10):
    h1 = w11 * x1 + w21 * x2
    h2 = w12 * x1 + w22 * x2

    o5 = sigmoid(w31 * h1 + w41 * h2)
    o6 = sigmoid(w32 * h1 + w42 * h2)

    delta5 = (o5 - y5) * o5 * (1 - o5)
    delta6 = (o6 - y6) * o6 * (1 - o6)

    delta1 = ((delta5 * w31) + (delta6 * w32)) * h1 * (1 - h1)
    delta2 = ((delta5 * w41) + (delta6 * w42)) * h2 * (1 - h2)

    learning_rate = 0.4
    alpha = 0.5

    w11 = alpha * w11 + learning_rate * delta1 * x1
    w21 = alpha * w21 + learning_rate * delta1 * x2
    w12 = alpha * w12 + learning_rate * delta2 * x1
    w22 = alpha * w22 + learning_rate * delta2 * x2
    w31 = alpha * w31 + learning_rate * delta5 * o5
    w41 = alpha * w41 + learning_rate * delta6 * o5
    w32 = alpha * w32 + learning_rate * delta5 * o6
    w42 = alpha * w42 + learning_rate * delta6 * o6

    E = 0.5 * ((o5 - y5) + (o6 - y6))
    mse_values.append(E)
    epoch_values.append(epoch + 1)

    print(f"Epoch {epoch + 1}: Error = {E:.4f}")

plt.plot(epoch_values, mse_values, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs. Epochs')
plt.grid(True)
plt.show()
