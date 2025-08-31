# =============================================
# 2.4 Supervised Learning - Classification Models (Fixed for Multiclass)
# =============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from itertools import cycle

# -------------------------- 1) Load Dataset
df = pd.read_csv("data/heart_disease_clean.csv")

X = df.drop("target", axis=1)
y = df["target"]

classes = y.unique()
n_classes = len(classes)

# Binarize labels for multiclass ROC
y_bin = label_binarize(y, classes=classes)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------- 2) Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs", multi_class='auto'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}

# Training & Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # For ROC curve, use predicted probabilities if binary or multiclass
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        # SVM decision function fallback
        y_prob = model.decision_function(X_test)
    
    # Metrics - weighted average for multiclass
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "y_prob": y_prob
    }

    print(f"\n🔹 {name} Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

# -------------------------- 3) ROC Curve & AUC (Multiclass)
plt.figure(figsize=(8,6))
colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])

for name, res in results.items():
    if n_classes == 2:
        # Binary classification ROC
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"][:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    else:
        # Multiclass ROC (One-vs-Rest)
        y_test_bin = label_binarize(y_test, classes=classes)
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], res["y_prob"][:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linestyle='--',
                     label=f"{name} Class {classes[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc='lower right', fontsize=8)
plt.grid(True)
plt.show()

# -------------------------- 4) Summary Table
summary_df = pd.DataFrame({
    name: {
        "Accuracy": res["Accuracy"],
        "Precision": res["Precision"],
        "Recall": res["Recall"],
        "F1-score": res["F1-score"]
    } for name, res in results.items()
}).T

print("\n✅ Model Performance Summary:")
print(summary_df.round(3))
