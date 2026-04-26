import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Generate or Load Your Data
# Assume input 5D measurement data has shape (D1, D2, D3, D4, D5)
D1, D2, D3, D4, D5 = 12, 10, 8, 6, 4  # Example dimensions

# Assume output 3D sample data has shape (K, M, L)
K, M, L = 20, 16, 12  # Example dimensions

# Generate synthetic data
# X is the input with shape (num_samples, D1, D2, D3, D4, D5)
num_samples = 1000  # Example number of samples
X = np.random.rand(num_samples, D1, D2, D3, D4, D5)


# Y is the output with shape (num_samples, K, M, L)
Y = np.random.rand(num_samples, K, M, L)


print(type(X))
print(np.shape(X))
print(type(Y))
print(np.shape(Y))

# Step 2: Normalize the Data
# Flatten the input to 2D for normalization
X_flattened = X.reshape(num_samples, -1)
scaler_X = StandardScaler()
X_normalized = scaler_X.fit_transform(X_flattened).reshape(X.shape)

# Flatten the output to 2D for normalization
Y_flattened = Y.reshape(num_samples, -1)
scaler_Y = StandardScaler()
Y_normalized = scaler_Y.fit_transform(Y_flattened).reshape(Y.shape)

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=42)

print(type(X))
print(np.shape(X))
print(type(Y))
print(np.shape(Y))
sys.exit()

# Step 4: Define the Model
model = Sequential([
    # Input layer accepting 5D data
    Input(shape=(D1, D2, D3, D4, D5)),

    # Reshape to combine D4 and D5 into a single channel dimension
    Reshape((D1, D2, D3, D4 * D5)),

    # First 3D Convolutional Layer
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2)),

    # Second 3D Convolutional Layer
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2)),

    # Flatten to reduce dimensionality
    Flatten(),

    # Fully connected dense layers
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),

    # Output layer to reshape into (K, M, L)
    Dense(K * M * L, activation='linear'),
    Reshape((K, M, L))
])

# Step 5: Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Step 6: Train the Model
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

# Step 7: Evaluate the Model
loss, mae = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Optional: Plotting the Training History
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()
