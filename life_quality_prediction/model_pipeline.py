from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness, TSNE
from sklearn.model_selection import train_test_split
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


