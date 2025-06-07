# %%
# === SHAP Explanation ===
import shap

import importlib
import preprocessing.pipeline
importlib.reload(preprocessing.pipeline)
from preprocessing.pipeline import get_preprocessor

from sklearn.pipeline import Pipeline

model_pipeline = Pipeline([
  ("preprocessor", get_preprocessor()),
])

# Remove earlier set_output call if present here (none present now)
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
