# =====================
# Import Libraries
# =====================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# =====================

# =====================
# Load Data
# =====================
# X_train = np.load("Processed Data/X_train.npy")
# X_test = np.load("Processed Data/X_test.npy")
X_train = np.load("results/X_train_sla.npy")
X_test = np.load("results/X_test_sla.npy")
Y_train = np.load("Processed Data/Y_train.npy")
Y_test = np.load("Processed Data/Y_test.npy")
# =====================

# =====================
# Reshape for CNN
# =====================
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("CNN Input Shape:", X_train.shape)
# =====================

# =====================
# CNN Model
# =====================
model = Sequential()

# conv layer 1
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Conv Layer 2
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Flatten
model.add(Flatten())

# Dense Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# =====================
# Compiling Model
# =====================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
# =====================

# =====================
# Train Model
# =====================
history = model.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)
# =====================

# Evaluation
# =====================

from sklearn.metrics import classification_report, confusion_matrix

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

print("\nCNN Confusion Matrix SLA-FS:")
print(confusion_matrix(Y_test, y_pred))
print("\nCNN Classification Report With SLA-FS:")
print(classification_report(Y_test, y_pred))

# ======================
# Save Model
# ======================
model.save("models/cnn_model_sla-fs.h5")

print("\nCNN model saved successfully.")