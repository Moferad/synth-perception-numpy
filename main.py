import numpy as np
from data_loader import load_folder
from numpy_nn import train, predict

X, labels = load_folder("data/")
X = X.T # reshape to (3072,10)

# convert string labels to 0 and 1
label_map = {name: i for i, name in enumerate(np.unique(labels))}
Y = np.array([label_map[l] for l in labels]).reshape(1, -1)

print(X.shape)
print(labels)
print(label_map)

parameters, costs = train(X, Y, n_h=64, learning_rate=0.01, num_iterations=1000)

predictions = predict(X, parameters)
accuracy = np.mean(predictions == Y) * 100
print(f"Training accuracy: {accuracy}%")
print(f"Predictions: {predictions}")
print(f"True labels: {Y}")