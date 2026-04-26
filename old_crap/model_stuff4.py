import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Reshape, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



with np.load('path_data.npz', allow_pickle=True) as data:
        im_size = data['im_size']
        voxel_index = data['voxel_index']
        voxel_data = data['voxel_data']
        nNeutrons = data['nNeutrons']
        nAngles = data['nAngles']
        angles = data['angles']
        source_xs = data['source_xs']
        source_zs = data['source_zs']
        det_xs = data['det_xs']
        det_zs = data['det_zs']

ver=2
Y = np.load(f'A_data_{ver}.npy', allow_pickle=True)
X = np.load(f'B_data_{ver}.npy', allow_pickle=True)
with np.load(f'meta_{ver}.npz', allow_pickle=True) as data:
        num_samples = data['num_samples']
        wavelengths = data['wavelengths']


# Step 1: Generate or Load Your Data
# Assume input 5D measurement data has shape (D1, D2, D3, D4, D5)
# D1, D2, D3, D4, D5 = nNeutrons, nAngles, len(wavelengths), 3, 3  # Example dimensions
D1, D2, D3, D4, D5 = 16, 10, 7, 3, 3  # Example dimensions
print(type(X))
print(np.shape(X))
print(type(Y))
print(np.shape(Y))

# Assume output 3D sample data has shape (K, M, L)
K, M, L = 12, 12, 3  # Example dimensions

# Step 2: Normalize the Data
# Flatten and normalize X
X_flat = X.reshape(num_samples, -1)
scaler_X = StandardScaler()
X_norm = scaler_X.fit_transform(X_flat).reshape(X.shape)

# Flatten and normalize Y
Y_flat = Y.reshape(num_samples, -1)
scaler_Y = StandardScaler()
Y_norm = scaler_Y.fit_transform(Y_flat).reshape(Y.shape)

# Step 3: Split the Data
X_train, X_temp, Y_train, Y_temp = train_test_split(X_norm, Y_norm, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Step 4: Define the Model
model = Sequential([
    Input(shape=(D1, D2, D3, D4, D5)),
    Reshape((D1, D2, D3, D4 * D5)),  # Combine D4 and D5
    
    Conv3D(32, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling3D(pool_size=2),
    
    Conv3D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling3D(pool_size=2),
    
    Flatten(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    
    Dense(K * M * L, activation='linear'),
    Reshape((K, M, L))
])

# Step 5: Compile the Model with Adam Optimizer and Learning Rate Scheduler
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Define Callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Step 6: Train the Model
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=[lr_reduction, early_stopping]
)

# Step 7: Evaluate the Model
test_loss, test_mae = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Step 8: Plot Training History
import matplotlib.pyplot as plt

# Loss Plot
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# MAE Plot
plt.figure(figsize=(10,5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()