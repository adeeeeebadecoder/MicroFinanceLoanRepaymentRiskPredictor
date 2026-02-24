import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# LOAD DATA
df = pd.read_excel("MF Bank loan Dataset (1).xlsx")

# clean column names
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# remove leakage columns
df = df.drop(columns=["paid_off_time", "past_due_days"])

# SELECT FEATURES
features = [
    "principal",
    "terms",
    "age",
    "gender",
    "highest_education",
    "guarantor"
]

X = pd.get_dummies(df[features], drop_first=True)

# target variable
y = df["loan_status"].apply(lambda x: 1 if x == "PAIDOFF" else 0)

# FEATURE ENGINEERING
X["loan_burden"] = X["principal"] / X["terms"]
X["age_group"] = X["age"] // 10

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# BALANCE DATA
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# TRAIN MODEL 
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_res, y_res)

# CHECK ACCURACY
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# SAVE MODEL & COLUMNS
joblib.dump(model, "loan_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")

print("✅ Model saved successfully!")