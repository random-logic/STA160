import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# === Custom Transformers ===

class ColumnDropper(BaseEstimator, TransformerMixin):
  def __init__(self, columns_to_drop=[]):
    self.columns_to_drop = columns_to_drop

  def fit(self, X, y=None):
    self.fitted_ = True
    return self

  def transform(self, X):
    return X.drop(columns=self.columns_to_drop, errors='ignore')

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

class GarageYrBltBinner(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    self.fitted_ = True
    return self
  
  def transform(self, X):
    def bin_year(year):
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

class LotFrontageFiller(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.RELEVANT_FEATURES = ['LotArea', 'Neighborhood', 'Street', 'LotConfig']

  # We are assuming X is already one hot encoded here
  def fit(self, X, y=None):
    X_known = X[X['LotFrontage'].notnull()]

    # Use only relevant predictors
    X_train = pd.get_dummies(X_known[self.RELEVANT_FEATURES])
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

    # Use only relevant predictors
    X_test = pd.get_dummies(X_missing[self.RELEVANT_FEATURES])

    # Align columns in case of one-hot mismatch
    X_test = X_missing.reindex(columns=self.dummy_columns, fill_value=0)

    # Predict
    X.loc[X['LotFrontage'].isnull(), 'LotFrontage'] = self.model.predict(X_test)

    return X
  
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

# === Pipeline Creation ===
def get_preprocessor():
  return Pipeline(steps=[
    ('preprocessing', Pipeline(steps=[
      ("drop_id", ColumnDropper(columns_to_drop=["Id"])),
      ("cat_na_fill", CategoricalNaFiller(excluded_cols=['GarageYrBlt'])),
      ("num_na_fill", NumericalNaFiller(excluded_cols=['LotFrontage'])), # Example - MasVnrArea
      ("garage_bin", GarageYrBltBinner()),
      ("lotfrontage_fill", LotFrontageFiller()),
    ])),
    ('transformer', OneHotEncoderScaler()),
  ])