"""
EDA.py - Exploratory Data Analysis for the heart disease dataset.

Steps:
  1. Load the data and look at its basic structure
  2. Missing values (how many, and are they random?)
  3. Duplicate rows
  4. Class balance of the target
  5. Outliers (IQR rule) and impossible values
  6. "Table 1": compare each feature between the two classes
  7. Plots (saved in the figures/ folder)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

TARGET = "Heart Disease Status"
os.makedirs("figures", exist_ok=True)  # folder to save the plots


# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
df = pd.read_csv("heart_disease.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumn types:")
print(df.dtypes)

# Split the columns into numeric and categorical (excluding the target)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols and c != TARGET]

print("\nNumeric features:", numeric_cols)
print("Categorical features:", categorical_cols)


# ---------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------
missing = pd.DataFrame({
    "count": df.isna().sum(),
    "percent": (df.isna().mean() * 100).round(2),
})
print("\n--- Missing values ---")
print(missing.sort_values("percent", ascending=False))

# Is the missingness random? For each column with missing values,
# compare the disease rate when the value is missing vs. present.
# A big difference means the data is NOT missing at random, so the
# fact that it is missing carries information.
print("\n--- Disease rate: missing vs. present ---")
is_sick = (df[TARGET] == "Yes")
for col in df.columns:
    if df[col].isna().sum() == 0:
        continue
    rate_missing = is_sick[df[col].isna()].mean()
    rate_present = is_sick[df[col].notna()].mean()
    print(f"{col:25s} missing: {rate_missing:.2f}   present: {rate_present:.2f}")


# ---------------------------------------------------------------
# 3. Duplicate rows
# ---------------------------------------------------------------
print("\nNumber of duplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------------
# 4. Class balance
# ---------------------------------------------------------------
print("\n--- Class balance ---")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))
# If one class is ~80%, a model that always predicts it gets 80% accuracy.
# That is why we will use ROC-AUC, recall and F1 instead of accuracy.


# ---------------------------------------------------------------
# 5. Outliers (IQR rule) and impossible values
# ---------------------------------------------------------------
# IQR rule: a value is an outlier if it is below Q1 - 1.5*IQR
# or above Q3 + 1.5*IQR.
print("\n--- Outliers (IQR rule) ---")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col:25s} range allowed: [{lower:.1f}, {upper:.1f}]   outliers: {n_outliers}")

# Impossible values: things like blood pressure = 0 are data errors,
# not real patients. These should be treated as missing values.
# (CRP can genuinely be close to 0, so we only check strictly negative.)
print("\n--- Impossible values (zero or negative) ---")
for col in numeric_cols:
    if col == "CRP Level":
        n_bad = (df[col] < 0).sum()
    else:
        n_bad = (df[col] <= 0).sum()
    print(f"{col:25s} {n_bad}")

# Skewness: values near 0 mean a symmetric distribution.
print("\n--- Skewness ---")
print(df[numeric_cols].skew().round(2))


# ---------------------------------------------------------------
# 6. "Table 1": compare features between the two classes
# ---------------------------------------------------------------
# Numeric features -> mean (std) per class + Mann-Whitney U test
#   (Mann-Whitney does not assume a normal distribution)
# Categorical features -> chi-square test of independence
print("\n--- Table 1: numeric features ---")
rows = []
for col in numeric_cols:
    no = df.loc[df[TARGET] == "No", col].dropna()
    yes = df.loc[df[TARGET] == "Yes", col].dropna()
    p = stats.mannwhitneyu(no, yes).pvalue
    rows.append({
        "feature": col,
        "No: mean (sd)": f"{no.mean():.1f} ({no.std():.1f})",
        "Yes: mean (sd)": f"{yes.mean():.1f} ({yes.std():.1f})",
        "p-value": round(p, 4),
    })
table1_numeric = pd.DataFrame(rows)
print(table1_numeric.to_string(index=False))

print("\n--- Table 1: categorical features ---")
rows = []
for col in categorical_cols:
    counts = pd.crosstab(df[col], df[TARGET])  # rows = categories, cols = No/Yes
    p = stats.chi2_contingency(counts).pvalue
    rows.append({"feature": col, "p-value": round(p, 4)})
table1_categorical = pd.DataFrame(rows)
print(table1_categorical.to_string(index=False))

# Save Table 1 so you can put it in the paper
table1_numeric.to_csv("figures/table1_numeric.csv", index=False)
table1_categorical.to_csv("figures/table1_categorical.csv", index=False)


# ---------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------
# 7a. Box plot of every numeric feature, split by class
for col in numeric_cols:
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df, x=TARGET, y=col)
    plt.title(f"{col} by {TARGET}")
    plt.tight_layout()
    plt.savefig(f"figures/box_{col.replace(' ', '_')}.png")
    plt.close()

# 7b. Histogram of every numeric feature (to see the distribution shape)
df[numeric_cols].hist(bins=30, figsize=(12, 9))
plt.tight_layout()
plt.savefig("figures/histograms.png")
plt.close()

# 7c. Correlation heatmap between numeric features
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between numeric features")
plt.tight_layout()
plt.savefig("figures/correlation.png")
plt.close()

print("\nDone. Plots and tables saved in the figures/ folder.")
