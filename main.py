# ---- Heart Disease ML (clean, portfolio-ready) ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", None)

CSV_PATH = "heartDisease_2020_sampling.csv"   # <-- make sure this filename matches your CSV

# ----------------------------
# 1) Load
# ----------------------------
df = pd.read_csv(CSV_PATH)
print("Dataset loaded:", df.shape)
print("\nSample:\n", df.head())

# ----------------------------
# 2) Target -> numeric (0/1)
# ----------------------------
# Many versions have "Yes"/"No" for HeartDisease
if df["HeartDisease"].dtype == object:
    df["HeartDisease"] = df["HeartDisease"].map({"No": 0, "Yes": 1}).astype("Int64")
else:
    # If it's already numeric as strings, coerce to int
    df["HeartDisease"] = pd.to_numeric(df["HeartDisease"], errors="coerce").astype("Int64")

# ----------------------------
# 3) Basic cleaning & encoding
# ----------------------------
# Fill missing values BEFORE one-hot
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "HeartDisease"]
num_cols = [c for c in df.columns if c not in cat_cols + ["HeartDisease"]]

# fill numeric NaNs with median
df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))
# fill categorical NaNs with mode
for c in cat_cols:
    mode_val = df[c].mode(dropna=True)
    df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

# One-hot encode all remaining categorical features
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nAfter encoding:", df.shape)
print(df.head())

# ----------------------------
# 4) Train/Test split
# ----------------------------
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5) Models (balanced for class imbalance)
# ----------------------------
tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    class_weight="balanced"
)
rf = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=12,
    class_weight="balanced"
)

tree.fit(X_train, y_train)
rf.fit(X_train, y_train)
print("\nModels trained ✓")

# ----------------------------
# 6) Evaluation helper
# ----------------------------
def eval_model(name, model, Xte, yte):
    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    cm = confusion_matrix(yte, y_pred)
    print(f"\n{name} Results")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    return y_pred, cm, acc, prec, rec, f1

y_pred_tree, cm_tree, acc_t, prec_t, rec_t, f1_t = eval_model("Decision Tree", tree, X_test, y_test)
y_pred_rf,   cm_rf,   acc_r, prec_r, rec_r, f1_r = eval_model("Random Forest", rf, X_test, y_test)

# ----------------------------
# 7) Confusion matrices (save + show both)
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm_tree, display_labels=["No Disease", "Disease"]).plot(
    cmap="Blues", ax=axes[0], colorbar=False
)
axes[0].set_title("Decision Tree")
ConfusionMatrixDisplay(cm_rf, display_labels=["No Disease", "Disease"]).plot(
    cmap="Greens", ax=axes[1], colorbar=False
)
axes[1].set_title("Random Forest")
plt.suptitle("Confusion Matrices — Tree vs Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrices.png")  # <-- saved image you can open/download
plt.show()

# also show RF alone (some environments only render the first axes)
plt.figure(figsize=(5,4))
ConfusionMatrixDisplay(cm_rf, display_labels=["No Disease", "Disease"]).plot(cmap="Greens")
plt.title("Random Forest Confusion Matrix (solo)")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
plt.show()

print("\nSaved figures: confusion_matrices.png and confusion_matrix_rf.png")

# ----------------------------
# 8) Tree rules (explainability)
# ----------------------------
print("\nDecision Tree Rules (truncated):\n")
print(export_text(tree, feature_names=list(X.columns))[:2000])  # print first ~2000 chars

# ----------------------------
# 9) Comparison table
# ----------------------------
results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [acc_t, acc_r],
    "Precision": [prec_t, prec_r],
    "Recall": [rec_t, rec_r],
    "F1": [f1_t, f1_r]
})
print("\nModel Comparison:\n", results.round(3))

# Optional: top features from Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
print("\nTop 10 Features (Random Forest):\n", importances.round(4))

