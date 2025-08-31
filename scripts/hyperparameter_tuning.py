# =============================================
# 2.6 Hyperparameter Tuning
# =============================================

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =============================
# 1. Load Dataset
# =============================
df = pd.read_csv("data/heart_disease_clean.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 2. Define Models & Hyperparameter Grids
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto'),
    "Random Forest": RandomForestClassifier(random_state=42)
}

param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],  # lbfgs only supports l2
        "solver": ["lbfgs"]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
}

# =============================
# 3. GridSearchCV / RandomizedSearchCV
# =============================
best_models = {}
for name, model in models.items():
    print(f"\n🔹 Tuning {name}...")
    
    # Choose GridSearchCV for Logistic Regression, RandomizedSearchCV for Random Forest
    if name == "Logistic Regression":
        search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
    else:
        search = RandomizedSearchCV(model, param_grids[name], n_iter=10, cv=5, scoring='accuracy', random_state=42)
    
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    
    print(f"Best parameters for {name}: {search.best_params_}")

# =============================
# 4. Evaluate Optimized Models
# =============================
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} - Test Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))
