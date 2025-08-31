# =============================================
# 2.1 Data Preprocessing & Cleaning
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------- 1) Load dataset 
df = pd.read_csv("data/heart_disease.csv", header=None)

# Add column names (UCI dataset doesn't have headers)
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = column_names

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# -------------------------- 2) Handle Missing Values
# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert numeric columns properly
df = df.apply(pd.to_numeric, errors='ignore')

# Impute missing values
for col in df.columns:
    if df[col].dtype == 'object':  # categorical
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # numerical
        df[col].fillna(df[col].median(), inplace=True)

print("Missing values after cleaning:", df.isnull().sum().sum())


# -------------------------- 3) Encode Categorical Features (One-Hot Encoding)
categorical_cols = ['cp', 'thal', 'slope', 'restecg']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Shape after encoding:", df.shape)


# -------------------------- 4) Standardize Numerical Features
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

scaler = StandardScaler()   # You can swap with MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("Standardization complete.")

# -------------------------- 5) Exploratory Data Analysis (EDA)
# Histogram of numerical features
df[num_cols].hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Boxplots for numerical features vs target
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f"Boxplot of {col} by Heart Disease Target")
    plt.show()



    # Save cleaned dataset for later steps
df.to_csv("data/heart_disease_clean.csv", index=False)
print("✅ Cleaned dataset saved to data/heart_disease_clean.csv")
