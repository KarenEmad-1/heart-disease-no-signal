"""
model.py - Train and compare classification models.

Steps:
  1. Load the data and separate features (X) from the target (y)
  2. Build a preprocessing step (imputation + scaling + one-hot encoding)
  3. Define the models, including a "dummy" baseline
  4. Evaluate every model with stratified 5-fold cross-validation

Important: the preprocessing is put INSIDE a Pipeline. This way the
imputer and scaler are fitted only on the training folds, which
prevents data leakage (reviewers check for this).
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

TARGET = "Heart Disease Status"
RANDOM_STATE = 42  # fixed seed so results are reproducible


# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
df = pd.read_csv("heart_disease.csv")

y = (df[TARGET] == "Yes").astype(int)   # 1 = heart disease, 0 = no
X = df.drop(columns=[TARGET])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]


# ---------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------
# Numeric: fill missing values with the median, then standardize
#   (scaling matters for Logistic Regression and SVM, not for trees)
numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# Categorical: fill missing values with the word "Missing" so that
# "missing" becomes its own category (useful for Alcohol Consumption,
# which is ~26% missing), then one-hot encode.
categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_steps, numeric_cols),
    ("cat", categorical_steps, categorical_cols),
])


# ---------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------
# class_weight="balanced" makes the model pay more attention to the
# minority class (the sick patients) instead of always predicting "No".
models = {
    # Baseline: always predicts the most common class.
    # Every real model MUST beat this, otherwise it learned nothing.
    "Dummy (baseline)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                        random_state=RANDOM_STATE),
    "SVM": SVC(class_weight="balanced", random_state=RANDOM_STATE),
}


# ---------------------------------------------------------------
# 4. Cross-validation
# ---------------------------------------------------------------
# Stratified: every fold keeps the same 80/20 class ratio.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Metrics to report (accuracy alone is misleading with imbalanced classes)
scoring = {
    "ROC-AUC": "roc_auc",
    "PR-AUC": "average_precision",
    "Recall": "recall",        # = sensitivity: how many sick patients we catch
    "Precision": "precision",
    "F1": "f1",
    "Accuracy": "accuracy",
}

results = []
for name, model in models.items():
    # Full pipeline = preprocessing + model
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

    # Save "mean ± std" across the 5 folds for each metric
    row = {"Model": name}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[metric] = f"{values.mean():.3f} ± {values.std():.3f}"
    results.append(row)
    print(f"Finished: {name}")

results = pd.DataFrame(results)
print("\n--- Cross-validation results (mean ± std over 5 folds) ---")
print(results.to_string(index=False))

results.to_csv("model_results.csv", index=False)
# How to read this: ROC-AUC of 0.5 = random guessing.
# If no model is clearly above 0.5, the features do not predict the target.


# ---------------------------------------------------------------
# 5. Permutation test: is the model better than chance?
# ---------------------------------------------------------------
# Idea: shuffle the target labels randomly (this destroys any real link
# between features and target), retrain, and record the score. Repeat
# 100 times. If the score on the REAL labels is not clearly higher than
# the scores on SHUFFLED labels, the model learned nothing real.
#
# p-value = fraction of shuffled runs that scored as well as the real one.
#   small p-value (< 0.05) -> the model found a real signal
#   large p-value          -> the model is no better than chance
print("\n--- Permutation test (Logistic Regression, 100 shuffles) ---")
pipe = Pipeline([("preprocess", preprocess),
                 ("model", LogisticRegression(max_iter=1000))])

score, perm_scores, p_value = permutation_test_score(
    pipe, X, y,
    cv=cv,
    scoring="roc_auc",
    n_permutations=100,
    random_state=RANDOM_STATE,
)

print(f"Real AUC:      {score:.3f}")
print(f"Shuffled AUC:  {perm_scores.mean():.3f} (std {perm_scores.std():.3f})")
print(f"p-value:       {p_value:.3f}")
