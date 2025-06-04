# %%
# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_predict, KFold

import scipy.stats as stats

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib
import preprocessing.pipeline
importlib.reload(preprocessing.pipeline)
from preprocessing.pipeline import get_preprocessor

# %%
# === Import data ===
train = pd.read_csv("../data/train.csv")
X_test = pd.read_csv("../data/test.csv")

X_train = train.drop("SalePrice", axis=1)
y_train = train["SalePrice"]

# %%
# === Build preprocessing + modeling pipeline ===
# Define RMSE scorer
rmse_scorer = make_scorer(
  lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
  greater_is_better=False
)

# RidgeCV with RMSE scoring
ridge_cv = RidgeCV(
  alphas=np.logspace(-3, 3, 13),
  scoring=rmse_scorer,
  cv=KFold(n_splits=10, shuffle=True, random_state=42)
)

# Update model_pipeline
model_pipeline = Pipeline([
  ("preprocessor", get_preprocessor()),
  ("model", TransformedTargetRegressor(
    regressor=ridge_cv,
    func=np.log1p,
    inverse_func=np.expm1
  ))
])

# %%
# === Cross-validated Predictions for RMSE Check ===
from sklearn.base import clone

cv = KFold(n_splits=10, shuffle=True, random_state=42)
val_preds = cross_val_predict(model_pipeline, X_train, y_train, cv=cv)
val_rmse = np.sqrt(mean_squared_error(y_train, val_preds))
print(f"Validation RMSE (from cross_val_predict): {val_rmse:.2f}")

# Manual per-fold training RMSE
train_rmses = []

for train_idx, val_idx in cv.split(X_train):
  X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
  y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
  
  model_clone = clone(model_pipeline)
  model_clone.fit(X_tr, y_tr)
  y_tr_pred = model_clone.predict(X_tr)
  rmse = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
  train_rmses.append(rmse)

avg_train_rmse = np.mean(train_rmses)
print(f"Average Training RMSE (across CV folds): {avg_train_rmse:.2f}")

# %%
# === Check skewness of numeric features only ===
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
X_train[numeric_cols].skew().sort_values(ascending=False)

# %%
# === Fit on full training data ===
model_pipeline.fit(X_train, y_train)

# %%
# === Evaluate on training data ===
y_train_pred = model_pipeline.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Train R² Score: {train_r2:.4f}")

# %%
# === Plot Residuals ===
residuals = y_train - y_train_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted SalePrice")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted SalePrice")
plt.tight_layout()
plt.savefig("../fig/lr/residuals_vs_predicted.png")
plt.show()

# %%
# === Check normality of residuals
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.savefig("../fig/lr/qq_plot_residuals.png")
plt.show()

# %%
# === Predict on test data ===
y_test_pred = model_pipeline.predict(X_test)

# %%
# === Wrap in DataFrame ===
submission = pd.DataFrame({
  "Id": X_test["Id"],  # assuming 'Id' is in test data
  "SalePrice": y_test_pred
})

submission.head()

# %%
# === Save in submission file ===
submission.to_csv("../data/submission_lr.csv", index=False)

# %%