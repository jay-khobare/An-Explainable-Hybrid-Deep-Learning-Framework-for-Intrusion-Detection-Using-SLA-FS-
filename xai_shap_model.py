# Import Libraries
# ======================
import numpy as np
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
# ======================

# Load Data
# ======================
X_train = np.load("Processed Data/X_train.npy")
X_test = np.load("Processed Data/X_test.npy")
Y_train = np.load("Processed Data/Y_train.npy")
Y_test = np.load("Processed Data/Y_test.npy")

# 👉 LOAD FEATURE NAMES (VERY IMPORTANT)
# df = pd.read_csv("Merge Dataset/final_merged_dataset.csv")
# feature_names = df.drop('Label', axis=1).columns
feature_names = pd.read_csv("Merged Data/final_merged_dataset.csv").columns[:-1]
# ======================

# Load Model
# ======================
model = joblib.load("Models/rf_model.pkl")
# model = joblib.load("Models/xgb_model.pkl")
# ======================

# SHAP Explainer
# ======================
explainer = shap.TreeExplainer(model)
# ======================

# Use TEST DATA (NOT TRAIN)
# ======================
sample_size = 1000
X_sample = X_test[:sample_size]
# ======================

# Compute SHAP values
# ======================
shap_values = explainer.shap_values(X_sample)

# Handle 3D SHAP output (VERY IMPORTANT)
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]   # class 1 = Attack
# ======================

#  SHAP Summary Plot (BEST ONE)
# ======================
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score

# -------------------------------
# 🔹 ROC-AUC
# -------------------------------
roc_auc = roc_auc_score(Y_test, y_prob)

# -------------------------------
# SHAP Explainability Metrics 
# -------------------------------

# Fidelity (model alignment)
fidelity = roc_auc

# Stability (variation of SHAP values)
stability = 1 - np.std(shap_values)

# Normalize stability (0–1)
stability = max(0, min(1, stability))

# Sparsity (how many features are active)
threshold = np.mean(np.abs(shap_values))
sparsity = np.sum(np.abs(shap_values) > threshold) / shap_values.size

# -------------------------------
# Create Evaluation Matrix
# -------------------------------

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)

evaluation_matrix = pd.DataFrame([{
    "Method": "SHAP (XAI)",

    # Performance
    "Accuracy": round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1 Score": round(f1, 4),
    "ROC-AUC": round(roc_auc, 4),

    # Explainability
    "Fidelity": round(fidelity, 4),
    "Stability": round(stability, 4),
    "Sparsity": round(sparsity, 4),
    "Interpretability": "High",
    "Computational Cost": "High"
}])

# -------------------------------
# Display
# -------------------------------
print("\n=========== SHAP EVALUATION MATRIX ===========\n")
print(evaluation_matrix)
# ======================

# Confusion Matrix
# ======================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d')
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Safe','Attack'], yticklabels=['Safe','Attack'])
plt.title("SHAP (XAI) Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# ======================

# ======================
# 🔹 Classification Report
# ======================
# report = classification_report(Y_test, y_pred, target_names=["Safe", "Attack"])

# print("\n===========XAI SHAP CLASSIFICATION REPORT ===========\n")
# print(report)

# Convert classification report to dictionary
report_dict = classification_report(Y_test, y_pred, target_names=["Safe", "Attack"], output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

print("\n=========== SHAP (XAI) CLASSIFICATION REPORT (DATAFRAME) ===========\n")
print(report_df)

# Save classification report
report_df.to_csv("Results/SHAP_XAI_classification_report.csv")
report_df.to_excel("Results/SHAP_XAI_classification_report.xlsx")

print("\nSaved classification report → Results/SHAP (XAI) classification_report.xlsx")


# Save Model
# =====================
joblib.dump(explainer, "Models/xai_shap_model.pkl")

# Saving Model
# ======================
results = pd.DataFrame([{
    "Model":"SHAP (XAI)",
    "Acccuracy":accuracy,
    "Precision":precision,
    "Recall":recall,
    "F1":f1
}])
results.to_csv("Results/xai_shap_result.csv", index=False)



#  SHAP Bar Plot (Top Features)
# ======================
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    plot_type="bar"
)
# ======================

# Feature Importance
# ======================
shap_importance = np.abs(shap_values).mean(axis=0)

# If still not 1D → fix it
if len(shap_importance.shape) > 1:
    shap_importance = shap_importance.mean(axis=1)

print("SHAP shape:", shap_values.shape)
print("Importance shape:", shap_importance.shape)
print("Feature names:", len(feature_names))

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": shap_importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 SHAP Features:")
print(importance_df.head(10))
np.save("Results/shap_importance.npy", shap_importance)

# ======================
# Publication-Ready SHAP Bar Plot
# ======================
top_k = 10

# Convert to numpy (safe indexing)
feature_names = np.array(feature_names)

# Sort
sorted_idx = np.argsort(shap_importance)[::-1][:top_k]
top_features = feature_names[sorted_idx]
top_values = shap_importance[sorted_idx]

# Plot
plt.figure(figsize=(10,6))
bars = plt.barh(top_features, top_values)

# Styling
plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Top 10 Important Features (SHAP)", fontsize=14)

# Highest at top
plt.gca().invert_yaxis()

# Add value labels on bars
for i, v in enumerate(top_values):
    plt.text(v + 0.001, i, f"{v:.4f}", va='center')

plt.tight_layout()

# Save (VERY IMPORTANT for report)
# plt.savefig("results/shap_top10_features.png", dpi=300)

plt.show()
# ======================

importance_df.to_csv("results/shap_feature_importance_rf.csv", index=False)
# ======================