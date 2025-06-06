# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv('../data/train.csv')

# Get all numeric columns excluding the target
numeric_cols = df.select_dtypes(include='number').columns
predictors = numeric_cols.drop('SalePrice', errors='ignore')

# Plot each predictor against SalePrice
for col in predictors:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x=col, y='SalePrice', alpha=0.5, edgecolor=None)
    plt.title(f'{col} vs SalePrice')
    plt.xlabel(col)
    plt.ylabel('SalePrice')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
# %%
