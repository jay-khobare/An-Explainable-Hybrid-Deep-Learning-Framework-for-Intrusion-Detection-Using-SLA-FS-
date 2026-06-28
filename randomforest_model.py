# Importing Libraries
# ======================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
import joblib
# ======================

# Preprocessed Data Loading
# ======================
X_train = np.load("Results/X_train_sla.npy")
X_test = np.load("Results/X_test_sla.npy")
Y_train = np.load("Processed Data/Y_train.npy")
Y_test = np.load("Processed Data/Y_test.npy")
# ======================

# Initialize Model
# ======================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
# ======================

# Train Model
# ======================
rf_model.fit(X_train,Y_train)
# ======================

# Predictions
# ======================
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]
# ======================

# Evaluation [Multi-class Safe]
# ======================
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
# ======================
# ROC-AUC
# ======================
from sklearn.metrics import roc_auc_score

y_prob = rf_model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(Y_test, y_prob)

print("ROC-AUC:", roc_auc)
# ======================

# ======================
# FEATURE IMPORTANCE
# ======================
import pandas as pd
import matplotlib.pyplot as plt

# feature_names = pd.read_csv("Merged Data/final_merged_dataset.csv").drop('Label', axis=1).columns

feature_names = np.load("Results/selected_features.npy", allow_pickle=True)
importances = rf_model.feature_importances_

feat_imp = pd.Series(importances, index=feature_names)

print("\nTop 10 Important Features With SLA-FS:")
print(feat_imp.sort_values(ascending=False).head(10))
feat_imp.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Feature Importance (Random Forest) With SLA-FS")
plt.show()

# ======================
# BASELINE MODEL (Logistic Regression)
# ======================
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, Y_train)

lr_pred = lr_model.predict(X_test)

print("\n______________Logistic Regression SLA-FS Results:_______________")
print("Accuracy:", accuracy_score(Y_test, lr_pred))
print("Precision:", precision_score(Y_test, lr_pred))
print("Recall:", recall_score(Y_test, lr_pred))
print("F1 Score:", f1_score(Y_test, lr_pred))

results = pd.DataFrame([{
    "Model":"Logistic Regression With SLA-FS",
    "Acccuracy":accuracy_score,
    "Precision":precision_score,
    "Recall":recall_score,
    "F1":f1_score
}])

results.to_csv("Results/baseline_result_SLA-FS.csv", index=False)



# Print Results
# ======================
print("\n______________Random Forest SLA-FS Results:_______________")
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)

print("\n RandomForest SLA-FS Classification Report")
print(classification_report(Y_test,y_pred))
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
plt.title("Random Forest With SLA-FS Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# ======================

# Save Model
# =====================
joblib.dump(rf_model, "Models/rf_model_sla-fs.pkl")
# ======================

# Saving Model
# ======================
results = pd.DataFrame([{
    "Model":"Random Forest",
    "Acccuracy":accuracy,
    "Precision":precision,
    "Recall":recall,
    "F1":f1
}])

results.to_csv("Results/randomforest_SLA-FS_result.csv", index=False)
print("\n Model & Results are saved successfully")
# ======================


