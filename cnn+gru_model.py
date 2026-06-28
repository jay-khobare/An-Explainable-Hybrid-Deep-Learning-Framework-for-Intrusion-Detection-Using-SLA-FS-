# ======================
# Import Libraries
# ======================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================
# Load Data
# ======================
X_train = np.load("Results/X_train_sla.npy")
X_test = np.load("Results/X_test_sla.npy")

Y_train = np.load("Processed Data/Y_train.npy")
Y_test = np.load("Processed Data/Y_test.npy")

# ======================
# Fix Labels
# ======================
Y_train = (Y_train > 0).astype(int)
Y_test = (Y_test > 0).astype(int)

# ======================
# Reshape
# ======================
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print("Shape:", X_train.shape)

# ======================
# Model
# ======================
input_layer = Input(shape=(X_train.shape[1], 1))

# ===== CNN Block =====
x = Conv1D(64, 3, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

x = Conv1D(128, 3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

# ===== GRU Block =====
x = GRU(64, return_sequences=True)(x)
x = Dropout(0.3)(x)

# ===== Replace Attention with GAP (more stable) =====
x = GlobalAveragePooling1D()(x)

# ===== Dense Block =====
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)

# Output
output = Dense(1, activation='sigmoid')(x)

# Model
model = Model(inputs=input_layer, outputs=output)

# ======================
# Compile
# ======================
optimizer = Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# Callbacks (IMPORTANT)
# ======================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# ======================
# Train
# ======================
history = model.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

# ======================
# Evaluate
# ======================
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

y_pred = y_pred.flatten()
Y_test = Y_test.flatten()

print("\n===== Evaluation Report (Hybrid CNN-GRU + SLA-FS) =====")

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred))
print("Recall:", recall_score(Y_test, y_pred))
print("F1 Score:", f1_score(Y_test, y_pred))

cm = confusion_matrix(Y_test, y_pred)
print("\nHybrid With SLA-FS Confusion Matrix:\n", cm)

print("\nHybrid With SLA-FS Classification Report:\n")
print(classification_report(Y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe','Attack'],
            yticklabels=['Safe','Attack'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Hybrid Model (SLA-FS)")
plt.show()

# loss, acc = model.evaluate(X_test, Y_test)
# print("Final Hybrid DL Accuracy with SLA-FS:", acc)

# ======================
# Save Model
# ======================
model.save("models/hybrid_cnn_gru_SLA-FS.keras")