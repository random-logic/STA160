# %%
# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

import copy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib
import preprocessing.pipeline
importlib.reload(preprocessing.pipeline)
from preprocessing.pipeline import get_preprocessor

# === Additional imports for cross-validation ===
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

# XGBoost import
from xgboost import XGBRegressor

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
        regressor=XGBRegressor(
            n_estimators=100,
            random_state=42,
            tree_method="hist",
            verbosity=0,
            n_jobs=-1
        ),
        func=np.log1p,
        inverse_func=np.expm1
    ))
])

# %%
# === Hyperparameter Tuning with GridSearchCV ===
param_grid = {
  "model__regressor__n_estimators": [200], # 50, 100, 200
  "model__regressor__max_depth": [3], # 3, 6, 10
  "model__regressor__learning_rate": [0.1], # 0.01, 0.05, 0.1
  "model__regressor__subsample": [0.8], # 0.8, 1.0
  "model__regressor__colsample_bytree": [1.0], # 0.8, 1.0
  "model__regressor__min_child_weight": [3], # 1, 3, 5
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=KFold(n_splits=10, shuffle=True, random_state=42),
                           scoring="neg_root_mean_squared_error",
                           n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

# Compute RMSE on cross-validated predictions from the training set
cv_train_preds = cross_val_predict(grid_search, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
cv_train_rmse = np.sqrt(mean_squared_error(y_train, cv_train_preds))
print(f"Train RMSE (from CV predictions): {cv_train_rmse:.2f}")
print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")
print(f"Best parameters: {grid_search.best_params_}")

'''
Train RMSE (from CV predictions): 27166.83
Best parameters: {'model__regressor__colsample_bytree': 0.8, 'model__regressor__learning_rate': 0.1, 'model__regressor__max_depth': 3, 'model__regressor__min_child_weight': 1, 'model__regressor__n_estimators': 200, 'model__regressor__subsample': 0.8}
'''

# Use the best estimator from the search for further evaluation
model_pipeline = grid_search.best_estimator_

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
# === Residual Plot ===
import seaborn as sns

residuals = y_train - y_train_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_train_pred, y=residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted SalePrice")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted (XGBoost)")
plt.tight_layout()

residual_plot_dir = "../fig/xgb"
os.makedirs(residual_plot_dir, exist_ok=True)
plt.savefig(os.path.join(residual_plot_dir, "residuals_vs_predicted.png"))
plt.show()

# %%
# === QQ Plot ===
import scipy.stats as stats

plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals (XGBoost)")
plt.tight_layout()

qq_plot_path = os.path.join(residual_plot_dir, "qq_plot.png")
plt.savefig(qq_plot_path)
plt.show()

# %%
# === Check if overfitting ===
print(f"Best CV RMSE: {-grid_search.best_score_:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")

# %%
# === Feature Importances ===
print("\n=== Feature Importances ===")
model = model_pipeline.named_steps["model"].regressor_
feature_names = model_pipeline.named_steps["preprocessor"].named_steps["transformer"].get_feature_names_out()
importances = model.feature_importances_

# Create a DataFrame for plotting
feat_imp_df = pd.DataFrame({
  "Feature": feature_names,
  "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 8))
plt.barh(feat_imp_df["Feature"][:30][::-1], feat_imp_df["Importance"][:30][::-1])
plt.xlabel("Importance")
plt.title("Top 30 Feature Importances (XGBoost)")
plt.tight_layout()

output_dir = "../fig/xgb"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "feature_importances.png"))

plt.show()

# %%
# === Retrain using only relevant features (Importance > 0.001) ===
# Select relevant features based on importance threshold
relevant_features = feat_imp_df[feat_imp_df["Importance"] > 0.001]["Feature"].values
print(f"Selected {len(relevant_features)} relevant features with importance > 0.001")

# Transform and filter training data
model_pipeline.named_steps["preprocessor"].set_output(transform="pandas")
X_train_transformed = model_pipeline.named_steps["preprocessor"].transform(X_train)
X_train_filtered = X_train_transformed[relevant_features]

# Reuse the best XGBRegressor from grid search
filtered_model = copy.deepcopy(model_pipeline.named_steps["model"].regressor_)

# Fit on filtered data
filtered_model.fit(X_train_filtered, y_train)

# Evaluate training performance
y_train_pred_filtered = filtered_model.predict(X_train_filtered)
train_rmse_filtered = np.sqrt(mean_squared_error(y_train, y_train_pred_filtered))
train_r2_filtered = r2_score(y_train, y_train_pred_filtered)
print(f"Train RMSE (filtered features): {train_rmse_filtered:.2f}")
print(f"Train R² Score (filtered features): {train_r2_filtered:.4f}")

# %%
# Cross-validation RMSE on filtered data
from sklearn.model_selection import cross_val_predict
cv_preds_filtered = cross_val_predict(filtered_model, X_train_filtered, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
cv_rmse_filtered = np.sqrt(mean_squared_error(y_train, cv_preds_filtered))
print(f"Validation RMSE (CV) with filtered features: {cv_rmse_filtered:.2f}")

# %%
# === SHAP Explanation ===
import shap

model_pipeline.named_steps["preprocessor"].set_output(transform="pandas")
X_train_transformed = model_pipeline.named_steps["preprocessor"].transform(X_train)

# Initialize SHAP explainer using filtered model and matching data
explainer = shap.Explainer(filtered_model, X_train_filtered)

# Compute SHAP values
shap_values = explainer(X_train_filtered)

# Summary plot for global feature importance
shap.summary_plot(shap_values, X_train_filtered, show=False)

# Save the plot
shap_output_dir = "../fig/xgb"
os.makedirs(shap_output_dir, exist_ok=True)
plt.savefig(os.path.join(shap_output_dir, "shap_summary.png"))
plt.close()

# Optional: show SHAP values for first training instance
shap.plots.waterfall(shap_values[0])

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
submission.to_csv("../data/submission_xgb.csv", index=False)

# %%
