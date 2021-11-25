# Construïm 4 conjunts de dades per classificar:
# En primer lloc un conjunt linealment separable on afegim renou

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

classifiers = [
    MLPClassifier(hidden_layer_sizes=1, random_state=1, max_iter=4000),
    MLPClassifier(hidden_layer_sizes=2, random_state=1, max_iter=4000),
    MLPClassifier(hidden_layer_sizes=3, random_state=1, max_iter=4000),
    MLPClassifier(hidden_layer_sizes=(2, 3), random_state=1, max_iter=4000)
]


def generate_datasets():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=33, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    # En segon lloc un que segueix una distribució xor
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                           X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    # Els afegim a una llista juntament amb els seus noms per tal de poder iterar
    # sobre ells
    return [
        ("linear", linearly_separable),
        ("moons", make_moons(noise=0.3, random_state=33)),  # Tercer dataset
        ("circles", make_circles(noise=0.2, factor=0.5, random_state=33)),  # Darrer dataset
        ("xor", (X_xor, y_xor))
    ]


def clf_surface(X, y, clf, score, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.1  # paràmetres per determinar la densitat de punts de la graella
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Realitzam la predicció de tots els valors de la graella (predict_proba)
    # dibuixam la graella i la seva predicció com un contorn (plt.contourf)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8)
    # ax.title(label=clf)
    ax.text(xx.min() + 0.3, yy.min() + 0.3, clf.hidden_layer_sizes, size=15, horizontalalignment="left")
    ax.text(xx.max() - 0.3, yy.min() + 0.3, ("Accuracy: %.2f" % score), size=15, horizontalalignment="right")
    # dibuixam els punts usats per entrenar diferenciant la seva classe (c=y)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    # ax.show()


comparison = []
fig, axs = plt.subplots(4, 4, figsize=(15, 15))
i = 0
for dataset in generate_datasets():
    name = dataset[0]
    data = dataset[1]
    X = data[0]
    y = data[1]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

    j = 0
    for clf in classifiers:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        comparison.append({
            "dataset": name,
            "classifier": clf,
            "accuracy": score
        })
        clf_surface(X, y, clf, score, axs[i][j])
        j += 1
    i += 1

plt.show()
print(pd.DataFrame(comparison).to_string())
