# Importing Libraries
# ======================
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
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

# Initalize XGBoost
# ======================
xgb_model = XGBClassifier(
    n_estimators = 100,
    max_depth = 6,
    learning_rate = .1,
    subsample = .8,
    colsample_bytree = .8,
    random_state = 42,
    eval_metric = 'mlogloss',
)
# ======================

# Train Model
# ======================
xgb_model.fit(X_train,Y_train)
# ======================

# Presdictions
# ======================
y_pred = xgb_model.predict(X_test)
# ======================

# ======================
# FEATURE IMPORTANCE
# ======================
import pandas as pd
import matplotlib.pyplot as plt

# feature_names = pd.read_csv("Merged Data/final_merged_dataset.csv").drop('Label', axis=1).columns
feature_names = np.load("Results/selected_features.npy", allow_pickle=True)

importances = xgb_model.feature_importances_

feat_imp = pd.Series(importances, index=feature_names)

print("\nTop 10 Important Features with SLA-FS:")
print(feat_imp.sort_values(ascending=False).head(10))
feat_imp.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Feature Importance (XGBoost) With SLA-FS")
plt.show()

# Evaluation [Multi-class Safe]
# ======================
accuracy = accuracy_score(Y_test,y_pred)
precission = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')
f1 = f1_score(Y_test, y_pred, average="weighted")
# ======================

# Print Results
# ======================
print("\n______________XGboost SLA-FS Results:_______________")
print("Accuracy:",accuracy)
print("Precision:",precission)
print("Recall:",recall)
print("F1 Score:",f1)

print("\n XGBoost SLA-FS Classification Report")
print(classification_report(Y_test,y_pred))
# ======================

# Confusion Matrix
# =====================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("XGBoost SLA-FS Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# ======================

# Save Model
# =====================
joblib.dump(xgb_model, "Models/xgb_model_sla-fs.pkl")
# ======================

# Saving Model
# ======================
results = pd.DataFrame([{
    "Model":"XGBoost",
    "Acccuracy":accuracy,
    "Precision":precission,
    "Recall":recall,
    "F1":f1
}])

results.to_csv("Results/xgboost_SLA-FS_result.csv", index=False)
print("\n Model & Results are saved successfully")
# ======================