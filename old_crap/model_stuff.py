import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Conv2D, Flatten, Dense, Reshape
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input

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
A_data = np.load(f'A_data_{ver}.npy', allow_pickle=True)
B_data = np.load(f'B_data_{ver}.npy', allow_pickle=True)
with np.load(f'meta_{ver}.npz', allow_pickle=True) as data:
        num_samples = data['num_samples']
        wavelengths = data['wavelengths']

# Example sizes for A and B
N = im_size  # Dimensions for A
K, M, L = nNeutrons, nAngles, len(wavelengths)  # Dimensions for B

# B_data = np.empty((num_samples, nNeutrons, nAngles, len(wavelengths), 3, 3), dtype=float)
# A_data = np.empty((num_samples, im_size, im_size, 3), dtype=float)


################################
#prepare data:
print('preparing data')
# Normalize data (important for neural networks)
scaler_A = StandardScaler()
scaler_B = StandardScaler()

# Flatten for normalization, then reshape back
A_data_flat = A_data.reshape(num_samples, -1)
B_data_flat = B_data.reshape(num_samples, -1)

A_data_normalized = scaler_A.fit_transform(A_data_flat).reshape(num_samples, N, N, 3)
B_data_normalized = scaler_B.fit_transform(B_data_flat).reshape(num_samples, K, M, L, 3, 3)

# Split into training and testing sets
A_train, A_test, B_train, B_test = train_test_split(A_data_normalized, B_data_normalized, test_size=0.2, random_state=42)


print(K)
print(M)
print(L)
################################
# Define the model that maps B to A
print('defining model')
# Define the model
model = Sequential([
    # Input layer specifying the input shape
    Input(shape=(K, M, L, 3, 3)),  # Note: Shape is (depth, height, width, channels)

    # First 3D convolutional layer
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2)),

    # Second 3D convolutional layer
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D(pool_size=(2, 2, 2)),

    # Flatten the output
    Flatten(),

    # Fully connected layers
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),

    # Output layer to reshape into (N, N, 3)
    Dense(N * N * 3, activation='linear'),
    Reshape((N, N, 3))
])

print('compiling model')
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.summary()  # Print model architecture

################################
# Train the model
print('training model')
history = model.fit(B_train, A_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate on the test set
loss, mae = model.evaluate(B_test, A_test)
print(f"Mean Absolute Error on test set: {mae}")

################################
