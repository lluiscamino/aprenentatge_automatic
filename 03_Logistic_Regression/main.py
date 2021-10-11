import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=5)

# Tractament de les dades: Separació i estandaritzat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenament i predicció
clf = SGDClassifier(loss="perceptron", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
# clf = SGDClassifier(loss="log", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
clf.fit(X_train_scaled, y_train)
prediction = clf.predict(X_test_scaled)

# Avaluació
cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:\t", accuracy)
f1 = f1_score(y_test, prediction)
print("F1 score:\t", f1)

# Mostrar els resultats
w0 = clf.intercept_[0]
w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]
slope = -(w0 / w2) / (w0 / w1)
point = [0, -w0 / w2]

fig, ax = plt.subplots()
plt.figure(1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
ax.axline(xy1=point, slope=slope)
plt.show()
