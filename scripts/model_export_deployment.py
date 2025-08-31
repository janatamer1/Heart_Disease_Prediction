# =============================================
# 2.7 Model Export & Deployment
# =============================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

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
# 2. Create Pipeline
# =============================
# Pipeline includes preprocessing + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf_model', RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=2, random_state=42
    ))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# =============================
# 3. Save Pipeline
# =============================
model_filename = "heart_disease_model.pkl"
joblib.dump(pipeline, model_filename)
print(f"Model pipeline saved as '{model_filename}'")

# =============================
# 4. Load & Test Pipeline (Optional)
# =============================
loaded_pipeline = joblib.load(model_filename)
sample_pred = loaded_pipeline.predict(X_test[:5])
print("Sample predictions on test data:", sample_pred)
