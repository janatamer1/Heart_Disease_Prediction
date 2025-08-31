# ============================
# 2.3 Feature Selection Pipeline
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("data/heart_disease_clean.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Step 1: Feature Importance (Random Forest)
# -----------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n🔹 Feature Importance (Random Forest):")
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.show()

# -----------------------------
# Step 2: Recursive Feature Elimination (RFE)
# -----------------------------
log_reg = LogisticRegression(max_iter=1000, solver="liblinear")
rfe = RFE(log_reg, n_features_to_select=5)  # Select top 5 features
rfe.fit(X, y)

rfe_selected = pd.DataFrame({
    "Feature": X.columns,
    "Selected": rfe.support_,
    "Ranking": rfe.ranking_
}).sort_values(by="Ranking")

print("\n🔹 Recursive Feature Elimination (RFE) Results:")
print(rfe_selected)

# -----------------------------
# Step 3: Chi-Square Test
# -----------------------------
# Scale features to [0,1] for Chi2 test
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi2_selector = SelectKBest(score_func=chi2, k="all")
chi2_selector.fit(X_scaled, y)

chi2_scores = pd.DataFrame({
    "Feature": X.columns,
    "Chi2 Score": chi2_selector.scores_
}).sort_values(by="Chi2 Score", ascending=False)

print("\n🔹 Chi-Square Test Results:")
print(chi2_scores)

# -----------------------------
# Step 4: Final Feature Selection
# -----------------------------
# Take top 5 from each method
top_rf = feature_importance_df["Feature"].head(5).tolist()
top_rfe = rfe_selected[rfe_selected["Selected"]==True]["Feature"].tolist()
top_chi2 = chi2_scores["Feature"].head(5).tolist()

# Take union of features found important by all 3 methods
final_features = list(set(top_rf + top_rfe + top_chi2))

print("\n✅ Final Selected Features for Modeling:")
print(final_features)

# Save reduced dataset
df_selected = df[final_features + ["target"]]
df_selected.to_csv("data/heart_selected_features.csv", index=False)

print("\n💾 Saved reduced dataset as 'data/heart_selected_features.csv'")
