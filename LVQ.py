import numpy as np

class LVQ:
    def __init__(self, prototype_count_per_class=1, learning_rate=0.1):
        self.prototype_count_per_class = prototype_count_per_class
        self.learning_rate = learning_rate
        self.errors = []

    def fit(self, X, y, epochs=30):
        # Initialize prototypes randomly
        self.prototypes = {}
        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            prototypes_indices = np.random.choice(len(X_c), self.prototype_count_per_class, replace=False)
            self.prototypes[c] = X_c[prototypes_indices].astype(float)  # Ensure prototypes are float64

        # Training
        for epoch in range(epochs):
            error_count = 0
            for i, x in enumerate(X):
                # Find the nearest prototype
                min_distance = float('inf')
                nearest_prototype = None
                for c, prototypes in self.prototypes.items():
                    for prototype in prototypes:
                        distance = np.linalg.norm(x - prototype)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_prototype = prototype
                            class_winner = c

                # Update the nearest prototype
                if y[i] == class_winner:
                    nearest_prototype += self.learning_rate * (x - nearest_prototype)
                else:
                    nearest_prototype -= self.learning_rate * (x - nearest_prototype)
                    error_count += 1

            # Calculate and store the error for the epoch
            error_rate = error_count / len(X)
            self.errors.append(error_rate)

    def predict(self, X):
        y_pred = []
        for x in X:
            min_distance = float('inf')
            predicted_class = None
            for c, prototypes in self.prototypes.items():
                for prototype in prototypes:
                    distance = np.linalg.norm(x - prototype)
                    if distance < min_distance:
                        min_distance = distance
                        predicted_class = c
            y_pred.append(predicted_class)
        return y_pred

# Dataset
X = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
y = np.array([1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1])  

# Initialize and train the LVQ model
lvq = LVQ(prototype_count_per_class=1, learning_rate=0.1)
lvq.fit(X, y)

# Print errors for each epoch
print("Errors for each epoch:")
for epoch, error in enumerate(lvq.errors):
    print(f"Epoch {epoch+1}: Error = {error}")
