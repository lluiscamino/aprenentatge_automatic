import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from Perceptron import Perceptron

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.8,
                           random_state=8)

y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

perceptron = Perceptron(eta=0.00000001, n_iter=100)
perceptron.fit(X, y)
y_prediction = perceptron.predict(X)

#  Mostram els resultats
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)

minim = np.min(X[:, 0])
maxim = np.max(X[:, 0])

first_coordinate = (minim, (-perceptron.w_[0] - (perceptron.w_[1] * minim)) / perceptron.w_[2])
second_coordinate = (maxim, (-perceptron.w_[0] - (perceptron.w_[1] * maxim)) / perceptron.w_[2])

plt.plot([first_coordinate[0], second_coordinate[0]], [first_coordinate[1], second_coordinate[1]])

plt.figure(2)
plt.plot(perceptron.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()


print(perceptron.w_)
