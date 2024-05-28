#Hebbian Learning
import numpy as np

x1 = np.array([-1, -1, 1, 1])
x2 = np.array([-1, 1, -1, 1])
t_AND = np.array([-1, -1, -1, 1])
t_OR = np.array([-1, 1, 1, 1])

def Hebb_learning(x1, x2, t, w1, w2, b):
    for i in range(len(x1)):
        y = w1 * x1[i] + w2 * x2[i] + b
        if y != t[i]:
            w1 += t[i] * x1[i]
            w2 += t[i] * x2[i]
            b += t[i]
    return w1, w2, b

def AND(x1, x2, w1, w2, b):
    weighted_sum = w1 * x1 + w2 * x2 + b
    if weighted_sum >= 0:
        return 1
    else:
        return -1

def OR(x1, x2, w1, w2, b):
    weighted_sum = w1 * x1 + w2 * x2 + b
    if weighted_sum >= 0:
        return 1
    else:
        return -1

print("AND Gate Using Hebbian Learning\n")
w1, w2, b = 0, 0, 0
w1, w2, b = Hebb_learning(x1, x2, t_AND, w1, w2, b)

print("-1 AND -1:", AND(-1, -1, w1, w2, b))
print("-1 AND 1:", AND(-1, 1, w1, w2, b))
print(" 1 AND -1:", AND(1, -1, w1, w2, b))
print(" 1 AND 1:", AND(1, 1, w1, w2, b))

print("\nOR Gate Using Hebbian Learning\n")
w1, w2, b = 0, 0, 0
w1, w2, b = Hebb_learning(x1, x2, t_OR, w1, w2, b)

print("-1 OR -1:", OR(-1, -1, w1, w2, b))
print("-1 OR 1:", OR(-1, 1, w1, w2, b))
print(" 1 OR -1:", OR(1, -1, w1, w2, b))
print(" 1 OR 1:", OR(1, 1, w1, w2, b))

#Error correction/Adaline
x1 = np.array([1, 1, -1, -1])
x2 = np.array([1, -1, 1, -1])
t = np.array([1, 1, 1, -1])

weight_1 = 0.1
weight_2 = 0.1
bias = 0.1
learning_rate = 0.1
iteration = 0

while True:
    total_error = 0
    print(f"Epoch: {iteration + 1}\n")
    print(f"\tInput\tTarget\tYin\tError\tW1\tW2\tBias\tFinal Error\tTotal Error")
    
    for i in range(4):
        y_in = bias + weight_1 * x1[i] + weight_2 * x2[i]
        error = t[i] - y_in
        final_error = error ** 2
        total_error += final_error
        
        weight_1 += learning_rate * error * x1[i]
        weight_2 += learning_rate * error * x2[i]
        bias += learning_rate * error
        
        print(f"\t{x1[i], x2[i]}\t{t[i]}\t{y_in:.4f}\t{error:.4f}\t{weight_1:.4f}\t{weight_2:.4f}\t{bias:.4f}\t{final_error:.4f}\t{total_error:.4f}")
    
    iteration += 1
    print("\n")
    
    if total_error <= 2:
        break

print("\nFinal Weights and Bias:")
print(f"Weight 1: {weight_1:.4f}\nWeight 2: {weight_2:.4f}\nBias: {bias:.4f}")

import numpy as np

x1 = np.array([1, 1, -1, -1])
x2 = np.array([1, -1, 1, -1])
t = np.array([-1, -1, -1, 1])

weight_1 = 0.1
weight_2 = 0.1
bias = 0.1
learning_rate = 0.1
iteration = 0

while True:
    total_error = 0
    print(f"Epoch: {iteration+1}\n")
    print(f"\tInput\tTarget\tYin\tError\tW1\tW2\tBias\tFinal Error\tTotal Error")

    for i in range(4):
        y_in = bias + weight_1 * x1[i] + weight_2 * x2[i]
        error = t[i] - y_in
        final_error = error ** 2
        total_error += final_error
        weight_1 += learning_rate * error * x1[i]
        weight_2 += learning_rate * error * x2[i]
        bias += learning_rate * error
        print(f"\t{x1[i], x2[i]}\t{t[i]}\t{y_in:.4f}\t{error:.4f}\t{weight_1:.4f}\t{weight_2:.4f}\t{bias:.4f}\t{final_error:.4f}\t{total_error:.4f}")

    iteration += 1
    print("\n")
    if total_error <= 2:
        break

print("\nFinal Weights and Bias:")
print(f"Weight 1: {weight_1:.4f}\nWeight 2: {weight_2:.4f}\nBias: {bias:.4f}")

#Memory Based
import numpy as np

def euclidean_distance(vector1, vector2):
    return np.sum((vector1 - vector2) ** 2)

def memory_based(training_data, input_data):
    distance_output_map = {}
    for sample_input, sample_output in training_data:
        distance = euclidean_distance(input_data, sample_input)
        if distance not in distance_output_map or distance_output_map[distance] > sample_output:
            distance_output_map[distance] = sample_output
    min_distance = min(distance_output_map.keys())
    return distance_output_map[min_distance]

training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 0),
    (np.array([1, 0]), 0),
    (np.array([1, 1]), 1)
]

test_data = np.array([1, 0])
print(f"Test data: {test_data}")
result = memory_based(training_data, test_data)
print(f"Result: {result}")

training_data = [
    (np.array([0, 0, 0]), 0),
    (np.array([0, 0, 1]), 0),
    (np.array([0, 1, 0]), 0),
    (np.array([0, 1, 1]), 1),
    (np.array([1, 0, 0]), 0),
    (np.array([1, 1, 0]), 1),
    (np.array([1, 1, 1]), 0)
]

test_data = np.array([1, 0, 1])
print(f"Test data: {test_data}")
result = memory_based(training_data, test_data)
print(f"Result: {result}")

#Competitive Learning
import numpy as np

input_vector1 = np.array([1, 1, 0, 0])
input_vector2 = np.array([0, 0, 0, 1])
input_vector3 = np.array([0, 0, 1, 1])
input_vector4 = np.array([1, 1, 0, 0])

weights_y1 = np.array([0.2, 0.6, 0.5, 0.9])
weights_y2 = np.array([0.8, 0.4, 0.7, 0.3])

learning_rate = 0.6
inputs = [input_vector1, input_vector2, input_vector3, input_vector4]

def euclidean_distance_squared(vector1, vector2):
    return np.sum((vector1 - vector2) ** 2)

def SOM(input_vector, weights_y1, weights_y2):
    distance_y1 = euclidean_distance_squared(input_vector, weights_y1)
    distance_y2 = euclidean_distance_squared(input_vector, weights_y2)
    winning_neuron = 1 if distance_y1 < distance_y2 else 2
    
    print("Winning Neuron is:", winning_neuron)
    
    if winning_neuron == 1:
        weights_y1 = weights_y1 + learning_rate * (input_vector - weights_y1)
    else:
        weights_y2 = weights_y2 + learning_rate * (input_vector - weights_y2)
    
    return weights_y1, weights_y2

for i in range(4):
    print("Input:", inputs[i])
    weights_y1, weights_y2 = SOM(inputs[i], weights_y1, weights_y2)
    print("Updated weights for y1:", weights_y1)
    print("Updated weights for y2:", weights_y2)
    print("")

