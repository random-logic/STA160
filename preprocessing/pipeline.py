import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor

# === Custom Transformers ===

class CategoricalNaFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cat_cols] = X[self.cat_cols].fillna("None")
        return X

class GarageYrBltBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
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

class MasVnrAreaFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)
        return X

class LotFrontageFiller(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        missing_mask = X["LotFrontage"].isnull()
        if missing_mask.any():
            X_missing = X.loc[missing_mask].drop(columns=["LotFrontage"])
            preds = self.model.predict(X_missing)
            X.loc[missing_mask, "LotFrontage"] = preds
        return X

def get_pipeline(lot_frontage_imputer_model):
    return Pipeline(steps=[
        ("cat_na_fill", CategoricalNaFiller()),
        ("garage_bin", GarageYrBltBinner()),
        ("masvnr_fill", MasVnrAreaFiller()),
        #("lotfrontage_fill", LotFrontageFiller(model=lot_frontage_imputer_model)),
    ])