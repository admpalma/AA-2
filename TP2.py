import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score, pair_confusion_matrix
from sklearn.neighbors import NearestNeighbors

import tp2_aux
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
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

    score_metrics = ["silhouette_score",
                      "adjusted_rand_score",
                      "rand_score",
                      "pair_precision",
                      "pair_recall",
                      "pair_f1"]

    kmin = 2
    kmax = 8

    results_for_each_k = {}
    for metric in score_metrics:
        results_for_each_k[metric] = []

    labeled_rows = labels[labels[:, 1] > 0]
    labeled_values = labeled_rows[:, 1].astype(int)
    labeled_indices = labeled_rows[:, 0].astype(int)
    selected_features = df[['pca_a', 'pca_b', 'pca_c']]

    for k in range(kmin, kmax + 1):
        results = {}
        for metric in score_metrics:
            results[metric] = []

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(selected_features)
        labeled_trimmed_col = selected_features.filter(items=labeled_indices, axis=0)
        results["silhouette_score"].append(
            silhouette_score(selected_features, kmeans.predict(selected_features)))
        for metric in score_metrics:
            if metric != "silhouette_score":
                results[metric].append(globals()[metric](labeled_values, kmeans.predict(labeled_trimmed_col)))

        for metric in score_metrics:
            results_for_each_k[metric].append(max(results[metric]))

    maximize_score = "silhouette_score"
    best_k = np.argmax(results_for_each_k[maximize_score]) + kmin

    Kmean = KMeans(n_clusters=5)
    kmeans_labels = Kmean.fit_predict(selected_features).astype(int)
    tp2_aux.report_clusters(np.array(range(563)),
                            kmeans_labels,
                            "kmeans_report.html")

    diag = pd.DataFrame(results_for_each_k)
    diag.plot(kind='line')
    plt.title("Kmeans Metrics")
    plt.ylabel("score")
    plt.xlabel("k")
    plt.tight_layout()
    plt.savefig("Kmeans_metrics.png", dpi=300)
    plt.show()

    distances, _ = NearestNeighbors(n_neighbors=6).fit(selected_features).kneighbors(selected_features)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 5]
    distances = distances[::-1]
    plt.grid(axis="y")
    plt.ylabel("eps")
    plt.plot(distances)
    plt.tight_layout()
    plt.show()

    dbscan = DBSCAN(eps=0.4)
    dbscan_labels = dbscan.fit_predict(selected_features).astype(int)
    tp2_aux.report_clusters(np.array(range(563)),
                            dbscan_labels,
                            "dbscan_report.html")

    eps_min = 0.35
    eps_max = 0.60

    results_for_each_eps = {}
    for metric in score_metrics:
        results_for_each_eps[metric] = []

    for eps in np.linspace(eps_min, eps_max, 5):
        results = {}
        for metric in score_metrics:
            results[metric] = []

        DBscan = DBSCAN(eps=eps)
        labeled_trimmed_col = selected_features.filter(items=labeled_indices, axis=0)
        results["silhouette_score"].append(
            silhouette_score(selected_features, DBscan.fit_predict(selected_features)))
        for metric in score_metrics:
            if metric != "silhouette_score":
                results[metric].append(globals()[metric](labeled_values, DBscan.fit_predict(labeled_trimmed_col)))

        for metric in score_metrics:
            results_for_each_eps[metric].append(max(results[metric]))

    diag2 = pd.DataFrame(results_for_each_eps)
    diag2.plot(kind='line')
    plt.title("DBSCAN Metrics")
    plt.ylabel("score")
    plt.xlabel("eps")
    plt.tight_layout()
    plt.savefig("DBSCAN_metrics.png", dpi=300)
    plt.show()

    agglomerative = AgglomerativeClustering(linkage="ward", n_clusters=7)
    agglomerative_labels = agglomerative.fit_predict(selected_features).astype(int)
    tp2_aux.report_clusters(np.array(range(563)),
                            agglomerative_labels,
                            "agglomerative_report.html")

    X = np.array(selected_features)

    target_clusters = 6
    current_clusters = 1
    cluster_tree = [[] for _ in range(len(X))]
    clusters = [[i for i in range(len(X))]]
    while current_clusters != target_clusters:
        cluster_to_split = clusters.pop(np.argmax((len(cluster) for cluster in clusters)))

        points_to_split = X[cluster_to_split]
        kmeans = KMeans(n_clusters=2).fit(points_to_split)
        new_clusters = [[], []]
        for index, point, label in zip(cluster_to_split, points_to_split, kmeans.labels_):
            new_clusters[label].append(index)
            cluster_tree[index].append(label)

        clusters.extend(new_clusters)
        current_clusters += 1

    tp2_aux.report_clusters_hierarchical(np.array(range(563)),
                            cluster_tree,
                            "bissecting_kmeans_report.html")