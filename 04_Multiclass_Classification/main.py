import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

TEST_DATA_PERCENTAGE = 0.3


def plot_digits(digits):
    _, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(label)
    plt.show()


def flat_all_image_matrices(image_matrices):
    return np.array([flat_image_matrix(image_matrix) for image_matrix in image_matrices])


def flat_image_matrix(image_matrix):
    return np.reshape(image_matrix, -1)


def split_data(images, labels, test_size):
    return train_test_split(images, labels, test_size=test_size, random_state=1)


def train_and_predict(X_train, X_test, y_train):
    clf = SGDClassifier(loss="log", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def print_model_scores(y_test, prediction):
    cf_matrix = confusion_matrix(y_test, prediction)
    print(cf_matrix)
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy:\t", accuracy)
    f1 = f1_score(y_test, prediction, average="micro")
    print("F1 score:\t", f1)


digits = datasets.load_digits()

plot_digits(digits)
print("Num. images", len(digits.images))

digits.images = flat_all_image_matrices(digits.images)
X_train, X_test, y_train, y_test = split_data(digits.images, digits.target, TEST_DATA_PERCENTAGE)
prediction = train_and_predict(X_train, X_test, y_train)
print_model_scores(y_test, prediction)
