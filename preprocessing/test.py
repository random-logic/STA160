#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import importlib
import pipeline
importlib.reload(pipeline)
from pipeline import get_preprocessor


# In[3]:


# === Load data ===
X_train = pd.read_csv("../data/train.csv").drop(columns=["SalePrice"])
X_test = pd.read_csv("../data/test.csv")

# === Build preprocessing pipeline ===
preprocessing_pipeline = get_preprocessor()

# === Fit transform the training data ===
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
preprocessing_pipeline


# In[4]:


X_train_preprocessed.head()


# In[5]:


print(X_train_preprocessed.dtypes.value_counts())  # summary of dtype counts
print(X_train_preprocessed.dtypes)                 # full listing by column


# In[6]:


print("Missing values per column in test data:")
print(X_train_preprocessed.isnull().sum()[X_train_preprocessed.isnull().sum() > 0].sort_values(ascending=False))


# In[7]:


# === 4. Run preprocessing on test data ===
# NOTE: fit_transform if fitting (e.g., training data), transform for test set
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# === 5. Inspect output ===
print("✅ Preprocessing successful.")
print(f"Output shape: {X_test_preprocessed.shape}")
X_test_preprocessed.head()


# In[8]:


print("Missing values per column in test data:")
print(X_test_preprocessed.isnull().sum()[X_test_preprocessed.isnull().sum() > 0].sort_values(ascending=False))


# In[9]:


# Identify columns with missing values
missing_cols = X_test_preprocessed.columns[X_test_preprocessed.isnull().any()]

# Print data types of columns with missing values
print("\nData types of columns with missing values:")
print(X_test_preprocessed[missing_cols].dtypes)


# In[10]:


# Output rows with missing values and ensure all columns are shown
print("\nRows with missing values in those columns:")
with pd.option_context('display.max_columns', None):
	display(X_test_preprocessed[X_test_preprocessed[missing_cols].isnull().any(axis=1)])


# %%

# === LotFrontageFiller cross-validation experiment ===
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np

# Prepare data for LotFrontageFiller cross-validation
X_lot = X_train.copy()
y_lot = X_train["LotFrontage"].copy()

# Drop rows where LotFrontage is NaN
mask = y_lot.notnull()
X_lot = X_lot[mask]
y_lot = y_lot[mask]

# Apply preprocessing steps before LotFrontageFiller
from pipeline import ColumnDropper, CategoricalNaFiller, NumericalNaFiller, GarageYrBltBinner, OutlierRemover, SkewedFeatureTransformer, OneHotEncoderScaler
from sklearn.ensemble import RandomForestRegressor

preprocess_steps = Pipeline(steps=[
    ("drop_id", ColumnDropper(columns_to_drop=["Id"])),
    ("cat_na_fill", CategoricalNaFiller(excluded_cols=['GarageYrBlt'])),
    ("num_na_fill", NumericalNaFiller(excluded_cols=['LotFrontage'])),
    ("garage_bin", GarageYrBltBinner()),
    ("remove_outliers", OutlierRemover()),
    ("skewed_transform", SkewedFeatureTransformer()),
    ("encode_scale", OneHotEncoderScaler())
])

X_lot_preprocessed = preprocess_steps.fit_transform(X_lot)

# Get the model from LotFrontageFiller
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_lot_preprocessed, y_lot)

# Training RMSE
y_pred_train = model.predict(X_lot_preprocessed)
train_rmse = np.sqrt(mean_squared_error(np.log1p(y_lot), np.log1p(y_pred_train)))
print(f"Log Training RMSE: {train_rmse:.4f}")
train_r2 = model.score(X_lot_preprocessed, y_lot)
print(f"Training R^2: {train_r2:.4f}")

# Cross-validated RMSE
log_rmse_scores = cross_val_score(model, X_lot_preprocessed, np.log1p(y_lot),
                                  scoring="neg_root_mean_squared_error", cv=5)
log_rmse_scores = -log_rmse_scores
print(f"Log CV RMSE: {log_rmse_scores.mean():.4f} (+/- {log_rmse_scores.std():.4f})")
log_r2_scores = cross_val_score(model, X_lot_preprocessed, y_lot,
                                  scoring="r2", cv=5)
print(f"CV R^2: {log_r2_scores.mean():.4f} (+/- {log_r2_scores.std():.4f})")

# %%
# See what remains after outlier removal
# Preview columns after OutlierRemover step
partial_pipeline = Pipeline(preprocess_steps.steps[:5])  # up to and including OutlierRemover
X_partial = partial_pipeline.fit_transform(X_lot)
print("Remaining columns after OutlierRemover:")
print(X_partial.columns.tolist())


# %%
print(len(X_partial.columns.tolist()))

# %%
