

# %%
# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_predict

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
model_pipeline = Pipeline([
  ("preprocessor", get_preprocessor()),
  ("model", TransformedTargetRegressor(
    regressor=RidgeCV(alphas=np.logspace(-3, 3, 13)),
    func=np.log1p,
    inverse_func=np.expm1
  ))
])

# %%
# === Cross-validated Predictions for Train RMSE ===
cv_train_preds = cross_val_predict(model_pipeline, X_train, y_train, cv=10)
cv_train_rmse = np.sqrt(mean_squared_error(y_train, cv_train_preds))
print(f"Train RMSE (from CV predictions): {cv_train_rmse:.2f}")

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
submission.to_csv("../data/submission.csv", index=False)

# %%