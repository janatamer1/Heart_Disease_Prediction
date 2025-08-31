# =============================================
# 2.5 Unsupervised Learning - Clustering
# =============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("data/heart_disease_clean.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# 2. K-Means Clustering
# =============================
# Elbow method to determine K
inertia = []
k_values = range(1, 11)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_values, inertia, 'bo-')
plt.xlabel("Number of clusters K")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.grid(True)
plt.show()

# Choose K=2 or 3 based on elbow
k_opt = 2
kmeans = KMeans(n_clusters=k_opt, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Compare K-Means clusters with actual labels
ari_kmeans = adjusted_rand_score(y, kmeans_labels)
print(f"K-Means Adjusted Rand Index (ARI) with actual labels: {ari_kmeans:.3f}")

# =============================
# 3. Hierarchical Clustering
# =============================
# Linkage matrix for dendrogram
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=k_opt)
agg_labels = agg.fit_predict(X_scaled)

# Compare Agglomerative clusters with actual labels
ari_agg = adjusted_rand_score(y, agg_labels)
print(f"Hierarchical Clustering Adjusted Rand Index (ARI) with actual labels: {ari_agg:.3f}")
