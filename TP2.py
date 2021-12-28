import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score, pair_confusion_matrix
from sklearn.neighbors import NearestNeighbors

import tp2_aux
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix, parallel_coordinates


def pair_precision(labels_true, labels_pred):
    (_, fp), (_, tp) = pair_confusion_matrix(labels_true, labels_pred)
    return tp / (tp + fp)


def pair_recall(labels_true, labels_pred):
    _, (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    return tp / (tp + fn)


def pair_f1(labels_true, labels_pred):
    (_, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    return (2 * tp) / (2 * tp + fp + fn)


def pca_plot():
    per_var = pca.explained_variance_ratio_ * 100
    pca_features = [f"PC{i}" for i in range(1, 7)]
    plt.bar(x=range(1, 7), height=per_var, tick_label=pca_features)
    plt.ylabel("Explained Variance (%)")
    plt.xlabel("Principal Component")
    plt.title("PCA Variance Explained")
    plt.tight_layout()
    plt.show()


def scatter_matrix_plot():
    axes = scatter_matrix(df, figsize=(45, 30), diagonal="kde", c=[colors[int(c)] for c in labels[:, 1]])
    for i in range(df.shape[1] - 1):
        for j in range(df.shape[1] - 1):
            axes[i, j].set_xlim(-2.0, 2.0)
            if i != j:
                axes[i, j].set_ylim(-2.0, 2.0)
    for j in range(df.shape[1] - 1):
        axes[df.shape[1] - 1, j].set_xlim(-2.0, 2.0)
    for i in range(df.shape[1] - 1):
        axes[i, df.shape[1] - 1].set_ylim(-2.0, 2.0)
    plt.tight_layout()
    plt.show()


def parallel_coordinates_plot():
    plt.figure(figsize=(45, 30))
    parallel_coordinates(df, "Name", color=colors, alpha=0.5)
    plt.tight_layout()
    plt.show()


def drop_correlated_features():
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]
    df.drop(to_drop, axis=1, inplace=True)


if __name__ == '__main__':
    images = tp2_aux.images_as_matrix()
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
    correlations = df.corr()

    labels = np.loadtxt("labels.txt", delimiter=',')
    df["Name"] = labels[:, 1]

    colors = ["lightgray", "red", "green", "blue"]
    scatter_matrix_plot()

    parallel_coordinates_plot()

    drop_correlated_features()

    parallel_coordinates_plot()

    df.drop(["Name"], axis=1, inplace=True)

    score_measures = ["silhouette_score",
                      "adjusted_rand_score",
                      "rand_score",
                      "pair_precision",
                      "pair_recall",
                      "pair_f1"]

    maxvars = 3
    kmin = 2
    kmax = 8

    cols = list(df.columns)
    results_for_each_k = {}
    for score in score_measures:
        results_for_each_k[score] = []

    vars_for_each_k = {}

    maximize_score = "adjusted_rand_score"

    labeled_rows = labels[labels[:, 1] > 0]
    labeled_values = labeled_rows[:, 1].astype(int)
    labeled_indices = labeled_rows[:, 0].astype(int)

    for k in range(kmin, kmax + 1):
        selected_variables = []
        cols = list(df.columns)
        while len(selected_variables) < maxvars:
            results = {}
            for score in score_measures:
                results[score] = []

            for col in cols:
                scols = []
                scols.extend(selected_variables)
                scols.append(col)
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(df[scols])
                labeled_trimmed_col = df[scols].filter(items=labeled_indices, axis=0)
                results["silhouette_score"].append(silhouette_score(df[scols], kmeans.predict(df[scols])))
                for score in score_measures:
                    if score != "silhouette_score":
                        results[score].append(globals()[score](labeled_values, kmeans.predict(labeled_trimmed_col)))

            selected_var = cols[np.argmax(results[maximize_score])]
            selected_variables.append(selected_var)
            cols.remove(selected_var)

        for score in score_measures:
            results_for_each_k[score].append(max(results[score]))
        vars_for_each_k[k] = selected_variables

    best_k = np.argmax(results_for_each_k[maximize_score]) + kmin

    selected_features = df[vars_for_each_k[best_k]]

    Kmean = KMeans(n_clusters=best_k)
    kmeans_labels = Kmean.fit_predict(selected_features).astype(int)
    tp2_aux.report_clusters(np.array(range(563)),
                            kmeans_labels,
                            "kmeans_report.html")

    diag = pd.DataFrame(results_for_each_k)

    distances, _ = NearestNeighbors(n_neighbors=6).fit(selected_features).kneighbors(selected_features)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 5]
    plt.grid(axis="y")
    plt.ylabel("eps")
    plt.plot(distances)
    plt.tight_layout()
    plt.show()

    dbscan = DBSCAN(eps=0.37)
    dbscan_labels = dbscan.fit_predict(selected_features).astype(int)
    tp2_aux.report_clusters(np.array(range(563)),
                            dbscan_labels,
                            "dbscan_report.html")
