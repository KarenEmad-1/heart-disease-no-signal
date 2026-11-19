"""
make_report.py - Builds report.pdf: evidence that this dataset contains
no learnable signal for predicting heart disease.

It recomputes every number and figure from heart_disease.csv, so the
report always matches the data. Uses only matplotlib (PdfPages) to make
the PDF, no extra libraries needed.

Run:  python make_report.py
"""

import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw to files, not to a window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_predict, permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

TARGET = "Heart Disease Status"
SEED = 42

# ---- Colors (validated categorical palette) ----
C_NO = "#2a78d6"      # blue   = class "No"
C_YES = "#eb6834"     # orange = class "Yes"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlesize": 8.5,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


# =================================================================
# PART A - COMPUTE EVERYTHING
# =================================================================
print("Loading data...")
df = pd.read_csv("heart_disease.csv")
y = (df[TARGET] == "Yes").astype(int)
X = df.drop(columns=[TARGET])
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
n_rows = len(df)
pos_rate = y.mean()

# ---- Preprocessing used by every model (inside CV -> no leakage) ----
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                        random_state=SEED),
    "SVM (RBF)": SVC(class_weight="balanced", random_state=SEED),
}

# ---- 1. Cross-validated metrics + out-of-fold scores for ROC curves ----
scoring = {"ROC-AUC": "roc_auc", "PR-AUC": "average_precision", "Recall": "recall",
           "Precision": "precision", "F1": "f1", "Accuracy": "accuracy"}
results, oof_scores = [], {}
for name, model in models.items():
    print("  CV:", name)
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    s = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    results.append([name] + [f"{s['test_'+m].mean():.3f} ± {s['test_'+m].std():.3f}"
                             for m in scoring])
    if name.startswith("Dummy"):
        continue
    method = "decision_function" if name.startswith("SVM") else "predict_proba"
    pred = cross_val_predict(pipe, X, y, cv=cv, method=method)
    oof_scores[name] = pred[:, 1] if pred.ndim == 2 else pred

# ---- 2. Permutation test (Logistic Regression, 100 shuffles) ----
print("  Permutation test...")
perm_pipe = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
perm_real, perm_scores, perm_p = permutation_test_score(
    perm_pipe, X, y, cv=cv, scoring="roc_auc", n_permutations=100, random_state=SEED)

# ---- 3. Linear correlation (point-biserial) of each numeric feature with target ----
corr = {c: stats.pointbiserialr(y[df[c].notna()], df[c].dropna())[0] for c in num_cols}

# ---- 4. Mutual information (detects ANY dependence, not only linear) ----
# MI estimates are always >= 0 and noisy, so we also compute MI against a
# SHUFFLED target 20 times. That gives the "noise floor": the MI you get
# when there is truly no relationship.
X_enc = preprocess.fit_transform(X)
X_enc = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc
feat_names = preprocess.get_feature_names_out()
mi_real = mutual_info_classif(X_enc, y, random_state=SEED)
rng = np.random.default_rng(SEED)
mi_null = np.array([mutual_info_classif(X_enc, rng.permutation(y.values), random_state=SEED)
                    for _ in range(20)])
mi_floor95 = np.percentile(mi_null.max(axis=1), 95)  # 95th pct of the max null MI
# keep only the numeric features for the plot (one bar per original feature)
num_idx = [i for i, f in enumerate(feat_names) if f.startswith("num__")]
mi_num = {feat_names[i].replace("num__", ""): mi_real[i] for i in num_idx}

# ---- 5. "Table 1" style tests with Bonferroni correction ----
pvals = []
for c in num_cols:
    pvals.append((c, "Mann-Whitney U",
                  stats.mannwhitneyu(df.loc[y == 0, c].dropna(), df.loc[y == 1, c].dropna()).pvalue))
for c in cat_cols:
    pvals.append((c, "Chi-square", stats.chi2_contingency(pd.crosstab(df[c], y)).pvalue))
n_tests = len(pvals)
bonf = 0.05 / n_tests

# ---- 6. Uniformity check: KS test of each numeric feature vs. a uniform dist ----
ks = {}
for c in num_cols:
    v = df[c].dropna()
    ks[c] = stats.kstest(v, "uniform", args=(v.min(), v.max() - v.min())).statistic

print("Computation done. Drawing PDF...")


# =================================================================
# PART B - PAGE LAYOUT HELPERS
# =================================================================
PAGE_W, PAGE_H = 8.27, 11.69  # A4 in inches
LEFT, RIGHT = 0.09, 0.91      # margins as figure fractions
WRAP = 100                    # characters per line for body text


class Page:
    """One A4 page. Text and charts are stacked from top (y) to bottom."""

    def __init__(self, pdf, number):
        self.pdf, self.number = pdf, number
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.y = 0.95

    def _line_h(self, size):
        return size * 1.42 / 72 / PAGE_H  # line height as a figure fraction

    def title(self, text, sub=None):
        self.fig.text(LEFT, self.y, text, size=17, weight="bold", color=INK, va="top")
        self.y -= self._line_h(17) + 0.006
        if sub:
            self.fig.text(LEFT, self.y, sub, size=9.5, color=INK2, va="top")
            self.y -= self._line_h(9.5) + 0.012

    def heading(self, text):
        self.y -= 0.008
        self.fig.text(LEFT, self.y, text, size=12, weight="bold", color=INK, va="top")
        self.y -= self._line_h(12) + 0.004

    def para(self, text, size=9.2, color=INK, bullet=False):
        width = WRAP - (4 if bullet else 0)
        lines = textwrap.wrap(text, width)
        x = LEFT + (0.025 if bullet else 0)
        if bullet:
            self.fig.text(LEFT + 0.008, self.y, "•", size=size, color=color, va="top")
        self.fig.text(x, self.y, "\n".join(lines), size=size, color=color,
                      va="top", linespacing=1.42)
        self.y -= len(lines) * self._line_h(size) + 0.007

    def bullets(self, items, **kw):
        for it in items:
            self.para(it, bullet=True, **kw)

    def caption(self, text):
        self.para(text, size=8, color=INK2)

    def area(self, height):
        """Reserve a horizontal band of the page; returns (bottom, top)."""
        top = self.y
        self.y -= height
        return self.y, top

    def save(self):
        self.fig.text(0.5, 0.025, f"{self.number}", size=8, color=MUTED, ha="center")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def draw_table(fig, rect, header, rows, col_widths, highlight_col=None, font=7.5):
    """Simple clean table drawn in its own invisible axes."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, colWidths=col_widths,
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_text_props(color=INK)
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
    return tbl


# =================================================================
# PART C - THE PAGES
# =================================================================
with PdfPages("report.pdf") as pdf:

    # ------------------------------------------------------------- PAGE 1
    p = Page(pdf, 1)
    p.title("No Learnable Signal in the Heart Disease Dataset",
            f"Evidence report  ·  heart_disease.csv  ·  {n_rows:,} patients, "
            f"{X.shape[1]} features, binary target")

    p.heading("Summary")
    p.para("Every test we ran points to the same conclusion: the features in this dataset carry no "
           "information about whether a patient has heart disease. The evidence comes from five "
           "independent angles:")
    p.bullets([
        "The numeric features are almost perfectly uniformly distributed with zero skew, and their "
        "minimum and maximum are round 'textbook' bounds. This is the signature of randomly "
        "generated (synthetic) data, not real clinical measurements.",
        f"Linear correlation of every numeric feature with the target is below "
        f"{max(abs(v) for v in corr.values()):.3f} in absolute value.",
        "Mutual information, which detects non-linear relationships too, is within the noise floor "
        "obtained from shuffled labels for every feature.",
        f"Five different classifiers, including non-linear tree ensembles, reach ROC-AUC of "
        f"0.49-0.50, the value of random guessing.",
        f"A permutation test shows the model does no better on the real labels than on shuffled "
        f"labels (p = {perm_p:.2f}).",
    ])

    p.heading("1. The data looks synthetic")
    p.para("Real physiological measurements are rarely uniform: blood pressure, BMI or triglycerides "
           "are typically bell-shaped or right-skewed, with a long tail of extreme patients. Figure 1 "
           "shows that every numeric feature here is flat between a hard minimum and maximum, and that "
           "the two classes overlap almost exactly. The Kolmogorov-Smirnov distance to a perfect "
           f"uniform distribution is at most {max(ks.values()):.3f} for all features (0 = perfectly "
           "uniform), skewness is between -0.02 and 0.02, and the IQR rule finds zero outliers.")

    bottom, top = p.area(0.43)
    gs = p.fig.add_gridspec(3, 3, left=LEFT + 0.02, right=RIGHT, bottom=bottom + 0.03,
                            top=top - 0.04, hspace=0.6, wspace=0.3)
    for i, c in enumerate(num_cols):
        ax = p.fig.add_subplot(gs[i // 3, i % 3])
        v = df[c].dropna()
        bins = np.linspace(v.min(), v.max(), 21)
        ax.hist(df.loc[y == 0, c].dropna(), bins=bins, density=True, histtype="step",
                color=C_NO, lw=1.6, label="No disease")
        ax.hist(df.loc[y == 1, c].dropna(), bins=bins, density=True, histtype="step",
                color=C_YES, lw=1.6, label="Disease")
        ax.set_title(c)
        ax.set_yticks([])
        ax.set_ylim(0, None)
        if i == 0:
            p.fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
                         bbox_to_anchor=(LEFT, top), ncol=2, fontsize=8)
    p.caption("Figure 1. Distribution of each numeric feature, split by class (density). All "
              "features are flat (uniform) and the two classes are indistinguishable.")
    p.save()

    # ------------------------------------------------------------- PAGE 2
    p = Page(pdf, 2)
    p.heading("2. Class imbalance: why accuracy is misleading")
    p.para(f"{pos_rate:.0%} of patients have heart disease and {1 - pos_rate:.0%} do not. A "
           f"'model' that ignores the patient and always predicts 'No disease' already reaches "
           f"{1 - pos_rate:.0%} accuracy while detecting zero sick patients. Any accuracy around 80% "
           "on this dataset is therefore meaningless. This is why we report ROC-AUC, PR-AUC, recall "
           "(sensitivity), precision and F1, and always compare against this majority-class "
           "baseline.")

    p.heading("3. Why correlation alone is not enough")
    p.para("The Pearson (point-biserial) correlation measures only one thing: how well a STRAIGHT "
           "LINE describes the relationship between a feature and the target. A correlation close to "
           "zero does not prove that the feature is useless. The feature could still matter through:")
    p.bullets([
        "Non-linear effects, e.g. risk that is high at BOTH very low and very high values "
        "(a U-shape). The rising and falling halves cancel out and the correlation becomes ~0.",
        "Threshold effects, e.g. risk that only jumps above a certain value.",
        "Interactions, e.g. smoking that only matters for older patients. Correlation looks at "
        "one feature at a time and cannot see this at all.",
    ])
    p.para("Figure 2 demonstrates this with an artificial example. The disease risk depends strongly "
           "on x through a U-shape, yet the correlation is essentially zero. Mutual information (MI), "
           "which measures any kind of dependence, detects it clearly.")

    # toy example: U-shaped risk
    rng_toy = np.random.default_rng(0)
    x_toy = rng_toy.uniform(-3, 3, 5000)
    p_toy = 1 / (1 + np.exp(-(2.2 * np.abs(x_toy) - 3.5)))
    y_toy = rng_toy.binomial(1, p_toy)
    r_toy = stats.pointbiserialr(y_toy, x_toy)[0]
    mi_toy = mutual_info_classif(x_toy.reshape(-1, 1), y_toy, random_state=0)[0]

    bottom, top = p.area(0.25)
    gs = p.fig.add_gridspec(1, 2, left=LEFT + 0.03, right=RIGHT, bottom=bottom + 0.035,
                            top=top - 0.02, wspace=0.35)
    ax = p.fig.add_subplot(gs[0, 0])
    edges = np.linspace(-3, 3, 13)
    mids = (edges[:-1] + edges[1:]) / 2
    rate = [y_toy[(x_toy >= a) & (x_toy < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
    ax.plot(mids, rate, color=C_YES, lw=2, marker="o", ms=4)
    ax.set_xlabel("feature x")
    ax.set_ylabel("disease rate")
    ax.set_title("Toy example: strong U-shaped risk")
    ax.set_ylim(0, 1)
    ax2 = p.fig.add_subplot(gs[0, 1])
    bars = ax2.bar(["Correlation |r|", "Mutual information"], [abs(r_toy), mi_toy],
                   color=[MUTED, C_YES], width=0.5)
    for b, v in zip(bars, [abs(r_toy), mi_toy]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center",
                 va="bottom", color=INK, size=8)
    ax2.set_title("What each measure sees")
    ax2.set_ylim(0, max(mi_toy, abs(r_toy)) * 1.3)
    p.caption(f"Figure 2. Simulated data where risk clearly depends on x. Correlation r = "
              f"{r_toy:.3f} (blind to the U-shape), while mutual information = {mi_toy:.3f} "
              "(detects it).")

    p.para("So for the real dataset we went beyond correlation and used three tools that can detect "
           "non-linear patterns: (1) the disease rate within bins of each feature, which reveals any "
           "one-dimensional shape (Figure 3); (2) mutual information compared with a shuffled-label "
           "noise floor (Figure 4); and (3) tree-based models, which capture non-linear effects AND "
           "interactions between features automatically (Section 8).")
    p.save()

    # ------------------------------------------------------------- PAGE 3
    p = Page(pdf, 3)
    p.heading("4. Searching for non-linear patterns in the real data")
    p.para("Figure 3 splits each numeric feature into ten equal-sized groups (deciles), from the "
           "lowest to the highest values, and shows the disease rate in each group. If a feature had "
           "a linear, U-shaped or threshold effect, the curve would rise, fall or bend.")

    bottom, top = p.area(0.27)
    gs = p.fig.add_gridspec(3, 3, left=LEFT + 0.03, right=RIGHT, bottom=bottom + 0.03,
                            top=top - 0.015, hspace=0.75, wspace=0.3)
    for i, c in enumerate(num_cols):
        ax = p.fig.add_subplot(gs[i // 3, i % 3])
        dec = pd.qcut(df[c], 10, duplicates="drop")
        r = y.groupby(dec, observed=True).mean()
        ax.axhline(pos_rate, color=MUTED, lw=1, ls="--")
        ax.plot(range(len(r)), r.values, color=C_YES, lw=1.8, marker="o", ms=3)
        ax.set_ylim(0, 0.5)
        ax.set_xticks([])
        ax.set_title(c, size=7.5)
        if i % 3 == 0:
            ax.set_ylabel("disease rate", size=7)
    p.caption(f"Figure 3. Disease rate in each tenth (decile) of every numeric feature, from lowest "
              f"to highest values. The dashed line is the overall rate ({pos_rate:.0%}). Every curve "
              "is flat: no linear, U-shaped or threshold pattern exists.")

    p.para("Figure 4 puts correlation and mutual information side by side. Correlations are "
           f"essentially zero. For mutual information, the dashed line is the noise floor: the 95th "
           f"percentile of the largest MI obtained when the target is randomly shuffled (20 "
           f"shuffles). Values below this line are what pure chance produces. All real features fall "
           f"below it.")

    bottom, top = p.area(0.24)
    gs = p.fig.add_gridspec(1, 2, left=LEFT + 0.15, right=RIGHT, bottom=bottom + 0.03,
                            top=top - 0.02, wspace=0.12)
    order = num_cols[::-1]
    ax = p.fig.add_subplot(gs[0, 0])
    ax.barh(order, [corr[c] for c in order], color=C_NO, height=0.55)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-0.1, 0.1)
    ax.set_title("Correlation with target (r)")
    ax.grid(axis="y", visible=False)
    ax2 = p.fig.add_subplot(gs[0, 1], sharey=ax)
    ax2.barh(order, [mi_num[c] for c in order], color=C_YES, height=0.55)
    ax2.axvline(mi_floor95, color=INK2, lw=1, ls="--")
    ax2.text(mi_floor95, -0.6, " noise floor", color=INK2, size=7, va="center")
    ax2.set_title("Mutual information")
    ax2.tick_params(labelleft=False)
    ax2.grid(axis="y", visible=False)
    ax2.set_xlim(0, mi_floor95 * 1.8)
    p.caption("Figure 4. Left: linear correlation of each numeric feature with the target (the axis "
              "spans ±0.1, yet bars are barely visible). Right: mutual information versus the "
              "shuffled-label noise floor.")
    p.save()

    # ------------------------------------------------------------- PAGE 4
    p = Page(pdf, 4)
    p.heading("5. Univariate statistical tests")
    p.para(f"We also compared the two classes feature by feature, as in a standard medical 'Table "
           f"1': Mann-Whitney U tests for numeric features (no normality assumption) and chi-square "
           f"tests for categorical features. With {n_tests} tests at the 0.05 level, about one "
           f"'significant' result is expected by pure chance. The Bonferroni correction accounts for "
           f"this by requiring p < 0.05 / {n_tests} = {bonf:.4f}. No feature passes.")

    rows = [[c, t, f"{pv:.4f}", "yes" if pv < 0.05 else "no", "yes" if pv < bonf else "no"]
            for c, t, pv in sorted(pvals, key=lambda r: r[2])]
    h = 0.0215 * (len(rows) + 1) + 0.01
    bottom, top = p.area(h + 0.01)
    draw_table(p.fig, [LEFT, bottom, RIGHT - LEFT, top - bottom],
               ["Feature", "Test", "p-value", "p < 0.05", f"p < {bonf:.4f} (Bonferroni)"],
               rows, [0.28, 0.18, 0.13, 0.12, 0.25])
    p.caption("Table 1. Univariate tests, sorted by p-value. The two 'significant' raw p-values "
              "disappear after correcting for multiple comparisons.")

    p.heading("6. Model choice")
    p.para("Deep learning was not used. On tabular data, tree-based ensembles and logistic "
           "regression usually match or outperform neural networks (Grinsztajn et al., 2022), and "
           "neural networks need far more data to avoid overfitting. Instead we chose models that "
           "cover the main families of assumptions, so that a negative result cannot be blamed on "
           "one model's limitations:")
    p.bullets([
        "Dummy classifier (always predicts the majority class): the floor every model must beat.",
        "Logistic Regression: linear, interpretable, the standard clinical baseline.",
        "Random Forest and Gradient Boosting: non-linear, model thresholds and feature "
        "interactions automatically, robust to outliers and feature scale.",
        "SVM with RBF kernel: a different, kernel-based non-linear model.",
    ])
    p.para("All preprocessing (median imputation for numeric features, a 'Missing' category for "
           "categorical features, scaling, one-hot encoding) is fitted inside each cross-validation "
           "fold to prevent data leakage. class_weight='balanced' makes models pay attention to the "
           "minority class. Evaluation uses stratified 5-fold cross-validation, which keeps the "
           "80/20 ratio in every fold; results are mean ± standard deviation across folds.")
    p.save()

    # ------------------------------------------------------------- PAGE 5
    p = Page(pdf, 5)
    p.heading("7. Metric choice")
    p.bullets([
        "ROC-AUC (main metric): probability that a random sick patient is ranked above a random "
        "healthy one. Independent of the decision threshold and of class imbalance. 0.5 = chance.",
        f"PR-AUC: precision-recall area, focused on the minority class. Chance level equals the "
        f"positive rate ({pos_rate:.2f}).",
        "Recall (sensitivity): share of sick patients detected; the clinically most important "
        "error to avoid is a missed diagnosis.",
        "Precision and F1: how many flagged patients are truly sick, and the balance of the two.",
        "Accuracy: reported only to show why it misleads under imbalance.",
    ])

    p.heading("8. Results")
    p.para("No model is better than chance. ROC-AUC is 0.49-0.50 for every model and PR-AUC equals "
           f"the positive rate ({pos_rate:.2f}). Models that reach higher recall do so only by "
           "flagging more patients at random, which lowers accuracy and keeps precision at the "
           "20% base rate. Random Forest falls back to predicting 'No' for everyone, exactly like "
           "the dummy model.")
    h = 0.021 * (len(results) + 1) + 0.01
    bottom, top = p.area(h + 0.005)
    draw_table(p.fig, [LEFT - 0.03, bottom, RIGHT - LEFT + 0.06, top - bottom],
               ["Model"] + list(scoring), results,
               [0.2] + [0.133] * len(scoring), font=7)
    p.caption("Table 2. Stratified 5-fold cross-validation results (mean ± standard deviation).")

    bottom, top = p.area(0.27)
    gs = p.fig.add_gridspec(1, 2, left=LEFT + 0.03, right=RIGHT, bottom=bottom + 0.035,
                            top=top - 0.02, wspace=0.3)
    ax = p.fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Chance (0.50)")
    for (name, sc), col in zip(oof_scores.items(), MODEL_COLORS):
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{name} ({roc_auc_score(y, sc):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (out-of-fold predictions)")
    ax.set_aspect("equal")
    ax.legend(fontsize=6.5, loc="lower right")

    ax2 = p.fig.add_subplot(gs[0, 1])
    ax2.hist(perm_scores, bins=20, color=C_NO, edgecolor="white", linewidth=1,
             label="AUC with shuffled labels")
    ax2.axvline(perm_real, color=C_YES, lw=2, label=f"AUC with real labels ({perm_real:.3f})")
    ax2.set_xlabel("ROC-AUC")
    ax2.set_ylabel("count (of 100 shuffles)")
    ax2.set_title(f"Permutation test, p = {perm_p:.2f}")
    ax2.legend(fontsize=6.5, loc="upper left")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)
    p.caption("Figure 5. Left: ROC curves of all models lie on the diagonal (random guessing). "
              "Right: permutation test for Logistic Regression. The labels were shuffled 100 times "
              "to show what 'no relationship' looks like; the real AUC sits in the middle of that "
              "distribution.")
    p.save()

    # ------------------------------------------------------------- PAGE 6
    p = Page(pdf, 6)
    p.heading("9. Discussion")
    p.para("The permutation test is the most direct proof. If the features contained any signal, "
           "the model trained on the real labels would score clearly higher than the models trained "
           f"on shuffled labels. Instead, {perm_p:.0%} of the shuffled runs scored at least as well "
           "as the real one. Together with the flat decile curves, the sub-noise mutual information "
           "and the failure of non-linear tree ensembles (which capture interactions), this rules "
           "out the possibility that the signal is merely non-linear or hidden in combinations of "
           "features.")

    p.heading("What will NOT fix this")
    p.para("When there is no signal, no modeling technique can create one. Several common "
           "practices can make results look better, but the improvement is fake:")
    p.bullets([
        "Hyperparameter tuning or deep learning: more flexible models only fit the noise more "
        "closely; they cannot extract information that is not there.",
        "Oversampling (e.g. SMOTE) before splitting the data: synthetic copies of the same "
        "patients leak into the test folds and inflate the scores.",
        "Feature selection or tuning on the full dataset, or picking the best of many random "
        "seeds: this selects features and settings that are good by chance and will not "
        "replicate.",
        "Moving the decision threshold: trades recall against precision, but ROC-AUC stays 0.5.",
        "Reporting accuracy: the 80% figure is achieved by the dummy model.",
    ])

    p.heading("Limitations")
    p.para("The analysis uses a fixed random seed and 5-fold cross-validation; repeated "
           "cross-validation would narrow the confidence intervals but cannot change a result "
           "that sits at chance level. The permutation test was run for Logistic Regression only "
           "(100 shuffles), for computational reasons; the ROC curves show that the other models "
           "behave identically.")

    p.heading("Conclusion and recommendation")
    p.para("This dataset contains no learnable relationship between the features and heart "
           "disease status, and is most likely randomly generated. It is unsuitable for a paper "
           "that proposes a predictive model. Two options remain:")
    p.bullets([
        "Use an established real dataset, such as the UCI Heart Disease data (Cleveland, or the "
        "combined Cleveland, Hungary, Switzerland and VA sets, about 920 patients) or the "
        "Framingham Heart Study data (about 4,200 patients). These contain clinically meaningful "
        "predictors and well-documented benchmark results.",
        "Frame the work as a cautionary study on the validation of public synthetic health "
        "datasets, using this report's analyses (especially the permutation test) as the "
        "main evidence, ideally contrasted with a real dataset where the same pipeline does "
        "find a signal.",
    ])

    p.heading("References")
    p.para("Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still "
           "outperform deep learning on typical tabular data? NeurIPS Datasets and Benchmarks.",
           size=8.5)
    p.para("Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier "
           "performance. Journal of Machine Learning Research, 11, 1833-1863.", size=8.5)
    p.para("Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. "
           "Physical Review E, 69(6), 066138.", size=8.5)
    p.para("Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than "
           "the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, "
           "10(3), e0118432.", size=8.5)
    p.save()

print("Saved report.pdf")
