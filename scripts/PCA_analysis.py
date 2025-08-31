# =============================================
# 2.2 PCA (Principal Component Analysis)
# =============================================

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load the cleaned dataset
df = pd.read_csv("data/heart_disease_clean.csv")

# 2. Separate features (X) and target (y)
X = df.drop("target", axis=1)  # all columns except target
y = df["target"]

# 3. Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# 4. Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Print variance explained by each component
print("\nExplained Variance by Each Principal Component:\n")
for i, var in enumerate(explained_variance, start=1):
    print(f"PC{i}: {var:.4f} ({var*100:.2f}% variance)")

print("\nCumulative Variance Explained:\n")
for i, cum_var in enumerate(cumulative_variance, start=1):
    print(f"PC1 to PC{i}: {cum_var:.4f} ({cum_var*100:.2f}%)")

# 5. Plot cumulative variance
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.grid(True)
plt.show()

# 6. Scatter plot of first two components
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", edgecolor="k", s=40)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - First Two Components")
plt.show()
