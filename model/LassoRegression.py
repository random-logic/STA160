# %%
# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.model_selection import GridSearchCV

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
rmse_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))),
    greater_is_better=False
)

model_pipeline = Pipeline([
    ("preprocessor", get_preprocessor()),
    ("model", TransformedTargetRegressor(
        regressor=Lasso(),
        func=np.log1p,
        inverse_func=np.expm1
    ))
])

param_grid = {
    'model__regressor__alpha': [0.0001, 0.001, 0.01, 0.1]
}

# === Apply preprocessing manually (after removing high SalePrice outliers) ===
preprocessor = get_preprocessor().named_steps['preprocessing']
X_test = X_test.copy()
X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)
mask = X_train_clean.index
y_train_clean = y_train.loc[mask].reset_index(drop=True)
X_train_clean = X_train_clean.reset_index(drop=True)

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_clean, y_train_clean)
model_pipeline = grid_search.best_estimator_

# %%
# === Results from CV ===
print(f"Best hyperparameters: {grid_search.best_params_}")

# %%
# === Fit on full training data ===
model_pipeline.fit(X_train_clean, y_train_clean)

# %%
# === Evaluate on training data ===
y_train_pred = model_pipeline.predict(X_train_clean)
train_rmse = np.sqrt(mean_squared_error(np.log1p(y_train_clean), np.log1p(y_train_pred)))
train_r2 = r2_score(y_train_clean, y_train_pred)

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Train R² Score: {train_r2:.4f}")

# %%
# === Compare training vs validation RMSE ===
val_preds_cv = cross_val_predict(model_pipeline, X_train_clean, y_train_clean, cv=KFold(n_splits=10, shuffle=True, random_state=42))
val_rmse = np.sqrt(mean_squared_error(np.log1p(y_train_clean), np.log1p(val_preds_cv)))
val_r2 = r2_score(y_train_clean, val_preds_cv)

print(f"Validation RMSE (CV): {val_rmse:.2f}")
print(f"Validation R² Score (CV): {val_r2:.4f}")

# %%
# === Feature Importances (Lasso Coefficients) ===
feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
coefs = model_pipeline.named_steps['model'].regressor_.coef_

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs
}).sort_values(by='Coefficient', key=np.abs, ascending=False)

plt.figure(figsize=(10, 12))
plt.barh(coef_df['Feature'][:30][::-1], coef_df['Coefficient'][:30][::-1])
plt.xlabel("Coefficient Value")
plt.title("Top 30 Feature Importances (Lasso Regression Coefficients)")
plt.tight_layout()
os.makedirs("../fig/lasso", exist_ok=True)
plt.savefig("../fig/lasso/feature_importances.png")
plt.show()

# %%
# === Plot Residuals ===
residuals = y_train_clean - y_train_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted SalePrice")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted SalePrice (Lasso Regression)")
plt.tight_layout()
plt.savefig("../fig/lasso/residuals_vs_predicted.png")
plt.show()

# %%
# === QQ Plot of Residuals ===
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals (Lasso Regression)")
plt.savefig("../fig/lasso/qq_plot_residuals.png")
plt.show()

# %%
# === Predict on test data ===
y_test_pred = model_pipeline.predict(X_test_clean)

# %%
# === Wrap in DataFrame ===
submission = pd.DataFrame({
    "Id": X_test["Id"],
    "SalePrice": y_test_pred
})

# %%
# === Save in submission file ===
submission.to_csv("../data/submission_lasso.csv", index=False)

# %%
