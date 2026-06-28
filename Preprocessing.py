# Importing Libraries
# =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# =====================

# Loading Dataset
# =====================
df = pd.read_csv("Merged Data/final_merged_dataset.csv")
df.head()
print(df.head())
df.columns = df.columns.str.strip()
print(df.columns.tolist())
# =====================

# Basic Inspections
# =====================
df.shape
df.info()
df.isnull().sum()
# =====================

# Data Cleaning
# =====================
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

print(df.isnull().sum().sum())
print(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
# =====================

# Label Encoding
# =====================
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# df['Label']=le.fit_transform(df['Label'])
# Convert to binary: 0 = Benign, 1 = Attack
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)


print(df['Label'].unique())
print(df['Label'].dtype)
# =====================

# Spliting
# =====================
X=df.drop('Label', axis=1)
Y=df['Label']
# =====================

# Split first
from sklearn.model_selection import train_test_split

X = df.drop('Label', axis=1)
Y = df['Label']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.5,
    random_state=42,
    stratify=Y
)

# Scale AFTER split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)   # fit ONLY on train
X_test = scaler.transform(X_test)         # transform test

# Apply SMOTE ONLY on training data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

# Visualization
# =====================
# Before SMOTE
Y_train.value_counts().plot(kind='bar')
plt.title("Before SMOTE")
plt.xticks([0,1], ['Safe', 'Attack'])
plt.show()

# After SMOTE
pd.Series(Y_train_res).value_counts().plot(kind='bar')
plt.title("After SMOTE (Balanced Data)")
plt.xticks([0,1], ['Safe', 'Attack'])
plt.show()

# Save Processed Data
# =====================
np.save("Processed Data/X_train.npy", X_train_res)
np.save("Processed Data/X_test.npy",X_test)
np.save("Processed Data/Y_train.npy", Y_train_res)
np.save("Processed Data/Y_test.npy", Y_test)