import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


# === Custom Transformers ===
class OutlierRemover(BaseEstimator, TransformerMixin):
  def __init__(self, z_thresh=4.0, outlier_ratio=0.05):
    self.z_thresh = z_thresh
    self.outlier_ratio = outlier_ratio
    self.cols_to_keep_ = []

  def fit(self, X, y=None):
    X_numeric = X.select_dtypes(include=["number"])
    z_scores = np.abs((X_numeric - X_numeric.mean()) / X_numeric.std())
    outlier_fractions = (z_scores > self.z_thresh).mean()
    self.cols_to_keep_ = outlier_fractions[outlier_fractions < self.outlier_ratio].index.tolist()
    return self

  def transform(self, X):
    return X[self.cols_to_keep_ + X.select_dtypes(exclude=["number"]).columns.tolist()]

  def get_feature_names_out(self, input_features=None):
    return self.cols_to_keep_

class SkewedFeatureTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, skew_threshold=1.0):
    self.skew_threshold = skew_threshold
    self.skewed_cols_ = []

  def fit(self, X, y=None):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    skew_vals = X[numeric_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    self.skewed_cols_ = skew_vals[skew_vals > self.skew_threshold].index.tolist()
    return self

  def transform(self, X):
    X = X.copy()
    for col in self.skewed_cols_:
      X[col] = np.log1p(X[col])
    return X

  def get_feature_names_out(self, input_features=None):
    return input_features

class ColumnDropper(BaseEstimator, TransformerMixin):
  def __init__(self, columns_to_drop=[]):
    self.columns_to_drop = columns_to_drop

  def fit(self, X, y=None):
    self.fitted_ = True
    return self

  def transform(self, X):
    return X.drop(columns=self.columns_to_drop, errors='ignore')

  def get_feature_names_out(self, input_features=None):
    if input_features is None:
      return None
    return [f for f in input_features if f not in self.columns_to_drop]

class CategoricalNaFiller(BaseEstimator, TransformerMixin):
  def __init__(self, excluded_cols=[]):
    self.excluded_cols = excluded_cols

  def fit(self, X, y=None):
    self.fitted_ = True
    return self

  def transform(self, X):
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.difference(self.excluded_cols)
    X[cat_cols] = X[cat_cols].fillna("None")
    return X

  def get_feature_names_out(self, input_features=None):
    return input_features

class GarageYrBltBinner(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    self.fitted_ = True
    return self
  
  def transform(self, X):
    def bin_year(year):
      try:
        year = float(year)
      except (ValueError, TypeError):
        return "NoGarage"

      if pd.isnull(year):
        return "NoGarage"
      elif year < 1940:
        return "Before1940"
      elif year < 1960:
        return "1940-1959"
      elif year < 1980:
        return "1960-1979"
      elif year < 2000:
        return "1980-1999"
      elif year < 2010:
        return "2000-2009"
      else:
        return "2010+"
    
    X = X.copy()
    X["GarageYrBlt"] = X["GarageYrBlt"].apply(bin_year).astype("category")
    return X

  def get_feature_names_out(self, input_features=None):
    return input_features

class NumericalNaFiller(BaseEstimator, TransformerMixin):
  def __init__(self, excluded_cols=[]):
    self.excluded_cols = excluded_cols

  def fit(self, X, y=None):
    self.fitted_ = True
    return self

  def transform(self, X):
    X = X.copy()
    num_cols = X.select_dtypes(include=["number"]).columns.difference(self.excluded_cols)
    X[num_cols] = X[num_cols].fillna(0)
    return X

  def get_feature_names_out(self, input_features=None):
    return input_features

class LotFrontageFiller(BaseEstimator, TransformerMixin):
  # We are assuming X is already one hot encoded here
  def fit(self, X, y=None):
    X_known = X[X['LotFrontage'].notnull()]

    # Use all columns except 'LotFrontage'
    X_train = X_known.drop(columns=['LotFrontage'])
    self.dummy_columns = X_train.columns
    y_train = X_known['LotFrontage']

    # Train
    self.model = RandomForestRegressor(n_estimators=100, random_state=0)
    self.model.fit(X_train, y_train)

    self.fitted_ = True

    return self
  
  def transform(self, X):
    X = X.copy()
    X_missing = X[X['LotFrontage'].isnull()]

    # Use all columns except 'LotFrontage'
    X_test = X_missing.drop(columns=['LotFrontage'])

    # Predict
    X.loc[X['LotFrontage'].isnull(), 'LotFrontage'] = self.model.predict(X_test)

    return X

  def get_feature_names_out(self, input_features=None):
    return input_features
  
class OneHotEncoderScaler(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    self.num_scaler = StandardScaler()
    # Configure transformers to output DataFrames
    self.cat_encoder.set_output(transform="pandas")
    self.num_scaler.set_output(transform="pandas")
    self.cat_cols = []
    self.num_cols = []

  def fit(self, X, y=None):
    self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    self.cat_encoder.fit(X[self.cat_cols])
    self.num_scaler.fit(X[self.num_cols])
    self.fitted_ = True
    return self

  def transform(self, X):
    X_cat = self.cat_encoder.transform(X[self.cat_cols])
    X_num = self.num_scaler.transform(X[self.num_cols])
    return pd.concat([X_num, X_cat], axis=1)
  
  def get_feature_names_out(self, input_features=None):
    cat_feature_names = self.cat_encoder.get_feature_names_out(self.cat_cols)
    num_feature_names = self.num_cols
    return np.array(list(num_feature_names) + list(cat_feature_names))

# === Pipeline Creation ===
def get_preprocessor():
  return Pipeline(steps=[
    ('preprocessing', Pipeline(steps=[
      ("drop_id", ColumnDropper(columns_to_drop=["Id"])), # Test to see if it's overfitting the lot frontage, maybe the imputer is causing overfitting
      ("cat_na_fill", CategoricalNaFiller(excluded_cols=['GarageYrBlt'])),
      ("num_na_fill", NumericalNaFiller(excluded_cols=['LotFrontage'])), # Example - MasVnrArea
      ("garage_bin", GarageYrBltBinner()),
      ("remove_outliers", OutlierRemover()),
      ("skewed_transform", SkewedFeatureTransformer()),
    ])),
    ('transformer', OneHotEncoderScaler()),
    ("lotfrontage_fill", LotFrontageFiller()),
  ])