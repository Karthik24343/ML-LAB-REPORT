# Task 1: Load dataset
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
X = data.data[:100, [0, 2]]
y = data.target[:100]

# Task 2: Preprocessing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Task 3: Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Task 4: Train library perceptron
from sklearn.linear_model import Perceptron

model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
model.fit(X_train, y_train)

# Task 5: Predict
y_pred = model.predict(X_test)

# Task 6: Compare actual and predicted
from sklearn.metrics import accuracy_score

print("Actual:", y_test)
print("Predicted:", y_pred)
print("Library Accuracy:", accuracy_score(y_test, y_pred))

# Task 7: Perceptron from scratch
def train_scratch(X, y):
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(100):
        for x, target in zip(X, y):
            pred = int(np.dot(x, w) + b >= 0)
            error = target - pred
            w += 0.1 * error * x
            b += 0.1 * error

    return w, b

w, b = train_scratch(X_train, y_train)

scratch_pred = (np.dot(X_test, w) + b >= 0).astype(int)

print("Scratch Predictions:", scratch_pred)
print("Scratch Accuracy:", accuracy_score(y_test, scratch_pred))