# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# %%
# Load the dataset
df = pd.read_csv('../data/train.csv')

# %%
print(df.shape)

# %%
df.head()

# %%
# Check for missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing values:\n", missing)

# %%
# Print data types of columns with missing values
print("\nData types of columns with missing values:\n", df[missing.index].dtypes)

# %% [markdown]
# There are a lot of missing values, the majority of which are categorical variables. A quick inspection of the dataset description shows that NA implies that a particular feature is not present. A great way to handle these is to label these categories as None.

# %%
# Replace NaN with "None" in all categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
df[cat_cols] = df[cat_cols].fillna("None")

# %% [markdown]
# For numerical columns, we will investigate the underlying distribution and consult the dataset description to interpret the meaning behind these values before deciding how to handle these.

# %%
num_missing_cols = df[missing.index].select_dtypes(include=['number']).columns

# Plot distributions
for col in num_missing_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {col} (Missing: {df[col].isnull().sum()} values)')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../fig/eda/dist_missing_{col}.png")
    plt.show()
    plt.close()

# %% [markdown]
# Garage year built have missing values due to some houses lacking garages. Binning GarageYrBlt simplifies the variable by grouping similar garage ages into broader categories, making patterns easier to capture and interpret. This approach is especially useful when handling missing values, as you can assign a clear bin like "NoGarage" rather than imputing a potentially misleading year. It also reduces model complexity and guards against overfitting from too many unique years.
# 
# The MasVanArea corresponds with the MasVanType, and all NA values for the area corresponds with all NA values for the type. We assume NA to be the absence of the feature, so this would imply an area of 0.
# 
# Lot frontage will heavily depend on other predictor variables such as 1rstFlrSF. To estimate this, we will use Random Forest Regression. We will not include the target variable to avoid data leakage, as it introduces information from the outcome into the predictors.

# %%
# Correlation heatmap of top 4 predictors most correlated with LotFrontage (excluding SalePrice)
corr_matrix = df.corr(numeric_only=True)
top_corr_lotfrontage = corr_matrix['LotFrontage'].drop('SalePrice', errors='ignore').abs().sort_values(ascending=False).head(5)
top_features = top_corr_lotfrontage.index.tolist()

plt.figure(figsize=(8, 6))
selected_features = list(OrderedDict.fromkeys(top_features + ['LotFrontage']))
sns.heatmap(df[selected_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Top 4 Correlated Features with LotFrontage")
plt.tight_layout()
plt.savefig("../fig/eda/lotfrontage_top_corr.png")
plt.show()
plt.close()

# %%
