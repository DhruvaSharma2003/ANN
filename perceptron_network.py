import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

learning_rate = 0.1
weights = np.random.rand(X.shape[1])
bias = 0
epochs = 0

while True:
    all_correct = True
    for i in range(len(X)):
        x = X[i]
        target = y[i]
        
        activation = np.dot(x, weights) + bias
        
        prediction = 1 if activation >= 0 else 0
       
        error = target - prediction
        if error != 0:
            all_correct = False
            weights += learning_rate * error * x
            bias += learning_rate * error
    epochs += 1
    if all_correct:
        break

print(f"Training completed in {epochs} epochs.")

def plot_decision_boundary():
    w1, w2 = weights
    b = bias
    if w2 != 0:  
        m = -w1 / w2
        c = -b / w2
    else:
        m = np.inf  
        c = weights[0]  

    
    margin = 0.2
    x_min = X[:, 0].min() - margin
    x_max = X[:, 0].max() + margin

    y_min = m * x_min + c  
    y_max = m * x_max + c  

    
    class_0_data = X[y == 0]
    class_1_data = X[y == 1]

    plt.plot([x_min, x_max], [y_min, y_max], 'g-', label='Decision Boundary (AND)')
    
    plt.scatter(class_0_data[:, 0], class_0_data[:, 1], c='blue', marker='o', label='Class 0')
    plt.scatter(class_1_data[:, 0], class_1_data[:, 1], c='red', marker='x', label='Class 1')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron")
    plt.xlim(x_min, x_max)
    plt.ylim(X[:, 1].min() - margin, X[:, 1].max() + margin)  # Adjusted y-axis limits
    plt.legend()
    plt.show()

plot_decision_boundary()
