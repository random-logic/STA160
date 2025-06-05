# %%
# === Imports ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
# === Get data ===
# Load the dataset
df = pd.read_csv('../data/train.csv')

# %%
# === Check skewness of numeric features ===
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
skew_values = df[numeric_cols].skew().sort_values(ascending=False)
print("Skewness of numeric features:")
print(skew_values)

# %%
# === Boxplots for highly skewed features ===
high_skew_cols = skew_values[skew_values > 1].index

for col in high_skew_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col} (Skewness = {skew_values[col]:.2f})')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"../fig/eda/{col}_boxplot.png")
    plt.show()
    plt.close()

# %%
# === Log1p transform highly skewed features ===
df_log_transformed = df.copy()
for col in high_skew_cols:
    df_log_transformed[col] = np.log1p(df_log_transformed[col])

# Boxplots of transformed features
for col in high_skew_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_log_transformed[col])
    plt.title(f'Boxplot of Log-Transformed {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"../fig/eda/log_{col}_boxplot.png")
    plt.show()
    plt.close()

# %%
# === Pairplots of highly skewed features vs SalePrice before transformation ===
for col in high_skew_cols:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[col], y=df['SalePrice'])
    plt.title(f'{col} vs SalePrice')
    plt.xlabel(col)
    plt.ylabel('SalePrice')
    plt.tight_layout()
    plt.savefig(f"../fig/eda/{col}_vs_saleprice_before.png")
    plt.show()
    plt.close()

# %%
# === Pairplots of highly skewed features vs SalePrice after log1p transformation ===

# Create individual scatter plots for each log-transformed skewed feature vs SalePrice
for col in high_skew_cols:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df_log_transformed[col], y=df['SalePrice'])
    plt.title(f'Log({col}) vs SalePrice')
    plt.xlabel(f'Log({col})')
    plt.ylabel('SalePrice')
    plt.tight_layout()
    plt.savefig(f"../fig/eda/log_{col}_vs_saleprice_after.png")
    plt.show()
    plt.close()

# %%
# === Target variable distribution ===
plt.figure(figsize=(8, 5))
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.savefig("../fig/eda/saleprice_distribution.png")
plt.show()
plt.close()

# %%
# === Boxplot of target variable ===
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['SalePrice'])
plt.title('Boxplot of Sale Prices')
plt.xlabel('Sale Price')
plt.tight_layout()
plt.savefig("../fig/eda/saleprice_boxplot.png")
plt.show()
plt.close()

# %%
# === Check skewness after log1p transformation of target variable ===
df['SalePrice_log'] = np.log1p(df['SalePrice'])
log_skew = df['SalePrice_log'].skew()
print(f"Skewness after log1p transformation: {log_skew:.4f}")

# %%
# === Distribution of log-transformed SalePrice ===
plt.figure(figsize=(8, 5))
sns.histplot(df['SalePrice_log'], kde=True)
plt.title('Distribution of Log-Transformed Sale Prices')
plt.xlabel('Log(Sale Price)')
plt.ylabel('Frequency')
plt.savefig("../fig/eda/saleprice_log_hist.png")
plt.show()
plt.close()

# %%
# === Boxplot of log-transformed SalePrice ===
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['SalePrice_log'])
plt.title('Boxplot of Log-Transformed Sale Prices')
plt.xlabel('Log(Sale Price)')
plt.tight_layout()
plt.savefig("../fig/eda/saleprice_log_boxplot.png")
plt.show()
plt.close()

# %% [markdown]
# There is a right skew in sale prices (target variable). Applying log1p solves the skewness issue and visually it looks approximately normal.

# %%
