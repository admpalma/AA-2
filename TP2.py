import numpy as np
import pandas as pd

import tp2_aux
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix, parallel_coordinates


def pca_plot():
    per_var = pca.explained_variance_ratio_ * 100
    pca_features = [f"PC{i}" for i in range(1, 7)]
    plt.bar(x=range(1, 7), height=per_var, tick_label=pca_features)
    plt.ylabel("Explained Variance (%)")
    plt.xlabel("Principal Component")
    plt.title("PCA Variance Explained")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images = tp2_aux.images_as_matrix()
    images = VarianceThreshold().fit_transform(images)
    images = np.divide(images, 255)
    pca = PCA(6)
    trans_pca = pca.fit_transform(images)
    print(sum(pca.explained_variance_ratio_))

    pca_plot()

    tsne = TSNE(n_components=6, method="exact")
    trans_tsne = tsne.fit_transform(images)

    isomap = Isomap(n_components=6)
    trans_isomap = isomap.fit_transform(images)

    extracted_features = np.concatenate((trans_pca, trans_tsne, trans_isomap), axis=1)
    extracted_features = StandardScaler().fit_transform(extracted_features)
    features = [f"{extractor}_{feature_id}"
                for extractor in ["pca", "tsne", "isomap"]
                for feature_id in ["a", "b", "c", "d", "e", "f"]]

    df = pd.DataFrame(extracted_features, columns=features)
    labels = np.loadtxt("labels.txt", delimiter=',')
    df["Name"] = labels[:, 1]

    colors = ["lightgray", "red", "green", "blue"]
    scatter_matrix(df, figsize=(45, 30), diagonal="kde", c=[colors[int(c)] for c in labels[:, 1]])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(45, 30))
    parallel_coordinates(df, "Name", color=colors, alpha=0.5)
    plt.tight_layout()
    plt.show()
