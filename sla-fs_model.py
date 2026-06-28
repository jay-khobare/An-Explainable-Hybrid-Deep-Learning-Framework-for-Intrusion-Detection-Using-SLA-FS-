# Import Libraries
# ======================
import numpy as np
# ======================

# Load SHAP & LIME Results
# ======================
shap_vals = np.load("Results/shap_importance.npy")
lime_vals = np.load("Results/lime_importance.npy")
# ======================

# Normalize
# ======================
shap_norm = shap_vals/np.max(shap_vals)
lime_norm = lime_vals/np.max(lime_vals)
# ======================

# SLA-FS Score
# ======================
# Weighted Sum
# SHAP
alpha = .6
# LIME 
beta = .4

sla_score = alpha*shap_norm+beta*lime_norm
# ======================

# Rank Features
# ======================
top_k=30
sla_top_idx = np.argsort(sla_score)[-top_k:]
# ======================

# Save Results
# ======================
np.save("Results/sla-fs_score.npy",sla_score)
np.save("Results/sla-fs_top_features.npy",sla_top_idx)

print("SLA-FS Completed.")
print("Top Features: ",sla_top_idx)
# ======================

# Visualization
# ======================
import matplotlib.pyplot as plt
import pandas as pd

sla_score = np.load("Results/sla-fs_score.npy")

# plt.figure()
# plt.bar(range(len(sla_score)), sla_score)
# plt.title("SLA-FS Feature Importance")
# plt.xlabel("Features")
# plt.ylabel("Score")
# plt.show()
# Load feature names
df = pd.read_csv("Merged Data/final_merged_dataset.csv")
feature_names = df.drop('Label', axis=1).columns.tolist()

# Sort features by importance
sorted_idx = np.argsort(sla_score)[::-1]

# Select top K features (recommended for readability)
top_k = 20
top_idx = sorted_idx[:top_k]

top_scores = sla_score[top_idx]
top_features = [feature_names[i] for i in top_idx]

# Plot
plt.figure(figsize=(10,6))
plt.barh(top_features, top_scores)
plt.xlabel("Normalized Feature Importance Score (SLA-FS)")
plt.title("Top 20 Most Important Features Identified by SLA-FS")
for i, v in enumerate(top_scores):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center')
plt.gca().invert_yaxis()  # highest on top
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.show()
# plt.savefig("images/sla_fs_top_features.png", dpi=300)
# ======================

# Appling SLA-FS
# ======================
X_train = np.load("Processed Data/X_train.npy")
X_test = np.load("Processed Data/X_test.npy")

top_features = np.load("Results/sla-fs_top_features.npy")

X_train_sla = X_train[:,top_features]
X_test_sla = X_test[:,top_features]

np.save("Results/x_train_sla.npy", X_train_sla)
np.save("Results/x_test_sla.npy",X_test_sla)
np.save("Results/selected_features.npy", top_features)

print("SLA_FS Dataset Creatted: ", X_train_sla.shape)
# ======================
