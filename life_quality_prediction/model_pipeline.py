from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness, TSNE
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import umap
import pandas as pd
import numpy as np

df = pd.read_csv('processed_data.csv')

X_train, X_test = train_test_split(df, test_size = 0.2, random_state = 42)

# data compression
# pca
pca = PCA(n_components=5)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# print(np.sum(pca.explained_variance_ratio_))
#
# X_test_reconstructed = pca.inverse_transform(X_test_pca)
# print(mean_squared_error(X_test_reconstructed, X_test))
#
# print(trustworthiness(X_test, X_test_pca, n_neighbors=10))
#
# # umap
#
# umap_model = umap.UMAP(n_components=5, n_neighbors=10)
# X_train_umap = umap_model.fit_transform(X_train)
# X_test_umap = umap_model.transform(X_test)
#
# print(trustworthiness(X_test, X_test_umap, n_neighbors=10))

# visualization

# models
# baseline
# k=5
# labels = np.random.randint(0, k, size=len(X_train_pca))
#
# print(silhouette_score(X_train_pca, labels))
# print(davies_bouldin_score(X_train_pca, labels))
# print(calinski_harabasz_score(X_train_pca, labels))

# baseline on lower dim



# kmeans
# best_score = -1
# best_k = None
#
# for k in range(2, 15):
#     model = KMeans(n_clusters=k, random_state=42)
#     labels = model.fit_predict(X_train)
#
#     score = silhouette_score(X_train, labels)
#
#     if score > best_score:
#         best_score = score
#         best_k = k


# best_kmeans_orspace = KMeans(n_clusters=best_k, random_state=42)
# train_labels_orspace = best_kmeans_orspace.fit_predict(X_train)
#
# print(silhouette_score(X_train, train_labels_orspace))
# print(davies_bouldin_score(X_train, train_labels_orspace))
# print(calinski_harabasz_score(X_train, train_labels_orspace))

# kmeans on lower dim
# best_score = -1
# best_k = None
#
# for k in range(2, 15):
#      model = KMeans(n_clusters=k, random_state=42)
#      labels = model.fit_predict(X_train_pca)
#
#      score = silhouette_score(X_train_pca, labels)
#
#      if score > best_score:
#          best_score = score
#          best_k = k
#
# print(best_k)
#
# best_kmeans_lowdim = KMeans(n_clusters=best_k, random_state=42)
# train_labels_lowdim = best_kmeans_lowdim.fit_predict(X_train_pca)
#
# print(silhouette_score(X_train, train_labels_lowdim))
# print(davies_bouldin_score(X_train, train_labels_lowdim))
# print(calinski_harabasz_score(X_train, train_labels_lowdim))

# DBSCAN and DBSCAN on lower dim don't work because it gets only 1 cluster
# best_eps = 0.1
# best_score = -1
# best_samples = 5
#
# for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     for samples in [5, 10, 20, 30, 40, 50]:
#         dbscan = DBSCAN(eps=eps, min_samples=samples)
#         labels = dbscan.fit_predict(X_train_pca)
#
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#
#         if n_clusters < 2:
#             continue
#
#         score = silhouette_score(X_train_pca, labels)
#
#         if score > best_score:
#             best_score = score
#             best_eps = eps
#             best_samples = samples
#
# print(best_eps, best_samples)
#
# best_dbscan = DBSCAN(eps=best_eps, min_samples=best_samples)
# labels = best_dbscan.fit_predict(X_train_pca)
#
# print(silhouette_score(X_train_pca, labels))
# print(davies_bouldin_score(X_train_pca, labels))
# print(calinski_harabasz_score(X_train_pca, labels))

# agglomerative clustering
# best_n = 3
# best_score = -1
#
# for n in [3, 5, 7, 9, 11]:
#     agg = AgglomerativeClustering(n_clusters=n)
#     labels = agg.fit_predict(X_train)
#
#     score = silhouette_score(X_train, labels)
#
#     if score > best_score:
#         best_score = score
#         best_n = n
# best_agg = AgglomerativeClustering(n_clusters=best_n)
# labels = best_agg.fit_predict(X_train)
#
# print(best_n)
#
# print(silhouette_score(X_train, labels))
# print(davies_bouldin_score(X_train, labels))
# print(calinski_harabasz_score(X_train, labels))


# agglomerative clustering on lower dim

# best_n = 3
# best_score = -1
#
# for n in [3, 5, 7, 9, 11]:
#     agg = AgglomerativeClustering(n_clusters=n)
#     labels = agg.fit_predict(X_train_pca)
#
#     score = silhouette_score(X_train_pca, labels)
#
#     if score > best_score:
#          best_score = score
#          best_n = n
#
# best_agg = AgglomerativeClustering(n_clusters=best_n)
# labels = best_agg.fit_predict(X_train_pca)
#
# X_2d = PCA(n_components=2).fit_transform(X_train_pca)
#
# plt.scatter(X_2d[:,0], X_2d[:,1], c=labels)
# plt.title("Agglomerative Clustering (2D PCA view)")
# plt.show()

# print(best_n)
#
# print(silhouette_score(X_train_pca, labels))
# print(davies_bouldin_score(X_train_pca, labels))
# print(calinski_harabasz_score(X_train_pca, labels))

# gmm
# best_n = 3
# best_score = -1
#
# for n in [3, 5, 7, 9, 11]:
#     gmm = GaussianMixture(n_components=n)
#     labels = gmm.fit_predict(X_train)
#
#     score = silhouette_score(X_train, labels)
#
#     if score > best_score:
#         best_score = score
#         best_n = n
#
# best_gmm = GaussianMixture(n_components=best_n)
# labels = best_gmm.fit_predict(X_train)
# print(best_n)
# print(silhouette_score(X_train, labels))
# print(davies_bouldin_score(X_train, labels))
# print(calinski_harabasz_score(X_train, labels))

# gmm on lower dim

# best_n = 3
# best_score = -1
#
# for n in [3, 5, 7, 9, 11]:
#     gmm = GaussianMixture(n_components=n)
#     labels = gmm.fit_predict(X_train_pca)
#
#     score = silhouette_score(X_train_pca, labels)
#
#     if score > best_score:
#         best_score = score
#         best_n = n
#
# best_gmm = GaussianMixture(n_components=best_n)
# labels = best_gmm.fit_predict(X_train_pca)
# print(best_n)
# print(silhouette_score(X_train_pca, labels))
# print(davies_bouldin_score(X_train_pca, labels))
# print(calinski_harabasz_score(X_train_pca, labels))






