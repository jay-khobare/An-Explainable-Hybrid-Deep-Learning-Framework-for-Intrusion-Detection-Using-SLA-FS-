# Importing Libraries
# ======================
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
# ======================

# Load Data
# ======================
X_train = np.load("Processed Data/X_train.npy")
X_test = np.load("Processed Data/X_test.npy")
# ======================

# Load Model[Choose Only One]
# ======================
model = joblib.load("Models/rf_model.pkl")
# model = joblib.load("models/xgb_model.pkl")
# ======================

# LIME Explainer
# ======================
import pandas as pd

df = pd.read_csv("Merged Data/final_merged_dataset.csv")
feature_names = df.drop('Label', axis=1).columns.tolist()
class_names = [f"class{i}" for i in range(len(np.unique(np.load("Processed Data/Y_train.npy"))))]

explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    discretize_continuous=True,
    mode='classification'
)
# ======================


# Find an ATTACK sample
for idx in range(len(X_test)):
    pred = model.predict(X_test[idx].reshape(1, -1))[0]
    if pred == 1:   # Attack
        i = idx
        break

pred = model.predict(X_test[i].reshape(1, -1))[0]
proba = model.predict_proba(X_test[i].reshape(1, -1))[0]

print(f"\nPrediction: {pred} (0=Safe, 1=Attack)")
print(f"Probabilities: {proba}")


exp = explainer.explain_instance(
    X_test[i],
    model.predict_proba,
    labels=(0,1)   # VERY IMPORTANT
)

print("\n_____________LIME Explanation for Instance {}_____________".format(i))
print(exp.as_list())
# ======================

# Visualization
# ======================
exp.save_to_file("Results/lime_explanation.html")
import matplotlib.pyplot as plt

fig = exp.as_pyplot_figure()
plt.title(f"LIME Explanation (Instance {i})")
plt.tight_layout()
exp.as_list(label=1)
plt.show()

fig = exp.as_pyplot_figure(label=1)
# plt.title(f"LIME Explanation (Attack) - Instance {i}")
plt.title(f"LIME Explanation (Attack) - Instance {i} | Pred={pred}")
exp.as_list(label=1)
plt.show()
# plt.savefig("images/lime_attack_instance.png", dpi=300, bbox_inches='tight')
# ======================



# Global Importance(Aggregated)
lime_importance = np.zeros(X_train.shape[1])
num_samples = 100

for i in range(num_samples):
    exp = explainer.explain_instance(
        X_train[i],
        model.predict_proba
    )

    for class_idx in exp.as_map():
        for feature_idx, value in exp.as_map()[class_idx]:
            lime_importance[feature_idx] += abs(value)

lime_importance /= num_samples


# =========================================================
# 🔹 LIME Evaluation Matrix
# =========================================================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------------
# 🔹 Model Performance (same model)
# -------------------------------
Y_test = np.load("Processed Data/Y_test.npy")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, y_prob)

# -------------------------------
# 🔹 LIME Explainability Metrics
# -------------------------------

# Fidelity → how well LIME approximates model locally
# -------------------------------
# 🔹 Fidelity → corrected version
# -------------------------------
fidelity_scores = []

num_samples_eval = 50

for j in range(num_samples_eval):
    exp = explainer.explain_instance(
        X_test[j],
        model.predict_proba
    )

    # Get LIME weights for class 1 (Attack)
    lime_weights = dict(exp.as_map()[1])

    # Approximate LIME score (sum of contributions)
    lime_score = sum(lime_weights.values())

    # True model probability
    model_score = model.predict_proba(X_test[j].reshape(1, -1))[0][1]

    # Compare difference
    fidelity_scores.append(1 - abs(model_score - lime_score))
    # Clipping
    fidelity_scores.append(max(0, 1 - abs(model_score - lime_score)))

fidelity = np.mean(fidelity_scores)

# -------------------------------
# Stability → consistency of explanations
# -------------------------------
stability = 1 - np.std(lime_importance)
stability = max(0, min(1, stability))

# -------------------------------
# Sparsity → how many features used
# -------------------------------
non_zero_features = np.sum(lime_importance > 0)
sparsity = non_zero_features / len(lime_importance)

# -------------------------------
# Create Evaluation Matrix
# -------------------------------
lime_evaluation_matrix = pd.DataFrame([{
    "Method": "LIME (XAI)",

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
    "Computational Cost": "Medium"
}])

# -------------------------------
# Display
# -------------------------------
print("\n=========== LIME (XAI) EVALUATION MATRIX ===========\n")
print(lime_evaluation_matrix)

# -------------------------------
# Save (for paper)
# -------------------------------
lime_evaluation_matrix.to_csv("Results/lime_xai_evaluation_matrix.csv", index=False)
lime_evaluation_matrix.to_excel("Results/lime_xai_evaluation_matrix.xlsx", index=False)

print("\nSaved → Results/lime_evaluation_matrix.xlsx")

np.save("Results/lime_importance.npy", lime_importance)

# print("LIME Global Importance Saved")
print(f"\nLIME Explanation for Instance {i}\n")

for feature, value in exp.as_list():
    direction = "↑ Attack" if value > 0 else "↓ Safe"
    print(f"{feature:40s} | Contribution: {value:.4f} ({direction})")
# ======================

# Save Model
# =====================
# joblib.dump(explainer, "Models/xai_lime_model.pkl")