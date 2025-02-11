import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def flat_all_image_matrices(image_matrices):
    return np.array([flat_image_matrix(image_matrix) for image_matrix in image_matrices])


def flat_image_matrix(image_matrix):
    return np.reshape(image_matrix, -1)


def principal_components_analysis(X):
    pca = PCA()
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    fig, ax = plt.subplots()
    ax.plot(cum_var_ratio)
    ax.set(xlabel='Components', ylabel='Cumulative variance')
    ax.grid()
    plt.show()


def gaussian_mixture_components_analysis(X):
    bics = []
    num_gauss = np.arange(30, 150, 10)
    for x in num_gauss:
        gm = GaussianMixture(n_components=x, random_state=0)
        gm.fit(X)
        bics.append(gm.bic(X))

    fig, ax = plt.subplots()
    ax.plot(num_gauss, bics)
    ax.set(xlabel='Components', ylabel='BIC')
    ax.grid()
    plt.show()


def plot_images(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
    plt.show()


digits = datasets.load_digits()
digits.images = flat_all_image_matrices(digits.images)
principal_components_analysis(digits.images)

pca = PCA(n_components=40)
X = pca.fit_transform(digits.images)

gaussian_mixture_components_analysis(X)

gm = GaussianMixture(n_components=50, random_state=0)
gm.fit(X)
samples, _ = gm.sample(100)

generated_images = [pca.inverse_transform(sample).reshape(8, 8) for sample in samples]
plot_images(generated_images)
