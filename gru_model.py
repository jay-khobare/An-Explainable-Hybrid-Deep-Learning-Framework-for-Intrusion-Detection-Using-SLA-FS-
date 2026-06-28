import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

X_train = np.load("Results/X_train_sla.npy")
X_test = np.load("Results/X_test_sla.npy")

Y_train = np.load("Processed Data/Y_train.npy")
Y_test = np.load("Processed Data/Y_test.npy")

# Convert to binary
Y_train = (Y_train > 0).astype(int)
Y_test = (Y_test > 0).astype(int)

# Reshape for GRU (IMPORTANT)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print("Shape:", X_train.shape)

# GRU Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

model = Sequential()

# GRU Layer
model.add(GRU(64, return_sequences=False, input_shape=(X_train.shape[1], 1)))

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))

# Output
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train GRU
history = model.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)


# ======================
# Evaluate Model
# ======================
y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

print("\nGRU Confusion Matrix With SLA-FS:")
print(confusion_matrix(Y_test, y_pred))

print("\nGRU Classification Report With SLA-FS:")
print(classification_report(Y_test, y_pred))

# ======================
# Save Model
# ======================
model.save("models/gru_model_sla-fs.h5")

print("\nGRU model saved successfully.")