import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for rendering
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from matplotlib.widgets import Button
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

path_ver = 6
ver = str(path_ver) + '0'
with np.load(f'data/path_data_{path_ver}.npz', allow_pickle=True) as data:
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


Y = np.load(f'data/A_data_{ver}.npy', allow_pickle=True)
# print(Y.shape)
# Y = Y[0:1000,:,:,:]
X = np.load(f'data/B_data_{ver}.npy', allow_pickle=True)
# print(X.shape)
# X = X[0:1000,:,:,:,:,:]

# print(X)
# print(Y)
with np.load(f'data/meta_{ver}.npz', allow_pickle=True) as data:
        num_samples = data['num_samples']
        wavelengths = data['wavelengths']

num_samples=250

# Step 1: Generate or Load Your Data
# Assume input 5D measurement data has shape (D1, D2, D3, D4, D5)
# D1, D2, D3, D4, D5 = nNeutrons, nAngles, len(wavelengths), 3, 3  # Example dimensions
# D1, D2, D3, D4, D5 = 16, 10, 7, 3, 3  # Example dimensions
D1, D2, D3, D4, D5 = int(nNeutrons), int(nAngles), int(len(wavelengths)), 3, 3  # Example dimensions

# print('*'*100)
# print(type(X))
# print(np.shape(X))
# print(type(Y))
# print(np.shape(Y))
# print(type(D1))
# print(type(int(nNeutrons)))
# print('*'*100)

# Assume output 3D sample data has shape (K, M, L)
# K, M, L = 4, 4, 3  # Example dimensions 
K, M, L = int(im_size), int(im_size), 3 
# Generate synthetic data
# X is the input with shape (num_samples, D1, D2, D3, D4, D5)
# num_samples = 1000  # Example number of samples
# X = np.random.rand(num_samples, D1, D2, D3, D4, D5)

# Y is the output with shape (num_samples, K, M, L)
# Y = np.random.rand(num_samples, K, M, L)

# Step 2: Normalize the Data
# Flatten the input to 2D for normalization
# X_flattened = X.reshape(num_samples, -1)
# scaler_X = StandardScaler()
# X_normalized = scaler_X.fit_transform(X_flattened).reshape(X.shape)
# we normalize "manually"



# Flatten the output to 2D for normalization
# Y_flattened = Y.reshape(num_samples, -1)
# scaler_Y = StandardScaler()
# Y_normalized = scaler_Y.fit_transform(Y_flattened).reshape(Y.shape)

# Min and Max values for input (X) and output (y)
X_min = X.min(axis=(0, 1, 2, 3), keepdims=True)  # Min across all dimensions
X_max = X.max(axis=(0, 1, 2, 3), keepdims=True)  # Max across all dimensions

Y_min = Y.min(axis=(0, 1, 2, 3), keepdims=True)  # Min across all dimensions
Y_max = Y.max(axis=(0, 1, 2, 3), keepdims=True)  # Max across all dimensions

# Min-Max normalization for input data
X_normalized = (X - X_min) / (X_max - X_min)

# Min-Max normalization for output data
Y_normalized = (Y - Y_min) / (Y_max - Y_min)

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=42)


# Save the scalers for later use
# joblib.dump(scaler_X, f'models/scaler_X_{ver}.joblib')
# joblib.dump(scaler_Y, f'models/scaler_Y_{ver}.joblib')
np.savez(f'models/scalers_{ver}.npz', X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max)

# print(type(X))
# print(np.shape(X))
# print(type(Y))
# print(np.shape(Y))

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

    # Third 3D Convolutional Layer
    Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
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

# Steep 4b: Use Gradual Warmup for Learning Rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Step 5: Compile the Model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# step 5b: callback for plotting and stopping
# Custom callback to plot the loss and stop the model when the stop button is pressed
class PlotLossesAndStop(Callback):
    def __init__(self, model_save_path='my_model_with_stop_button.h5'):
        super().__init__()
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.stop_training_flag = False
        self.model_save_path = model_save_path  # Path to save the model

        # Set up the figure and axis for the plot
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Add a stop button on the plot
        stop_ax = plt.axes([0.8, 0.05, 0.1, 0.075])  # Position of the stop button
        self.stop_button = Button(stop_ax, 'Stop')
        self.stop_button.on_clicked(self.stop_training)

        # Make sure the plot is interactive
        plt.ion()
        plt.show()

    def stop_training(self, event):
        """Callback to stop training when the button is clicked and save the model."""
        print("Stopping training...")
        self.stop_training_flag = True
        self.model.stop_training = True

        # Save the model as soon as training is stopped
        print(f"Saving the model to {self.model_save_path}")
        self.model.save(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        # Update the plot
        self.ax.clear()
        self.ax.plot(self.epochs, self.losses, label='Training Loss')
        self.ax.plot(self.epochs, self.val_losses, label='Validation Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()

        # Draw the updated plot and pause briefly to keep it responsive
        self.fig.canvas.draw()
        plt.pause(0.001)  # Short pause to ensure updates

        # Check if stop button was pressed
        if self.stop_training_flag:
            print(f"Training stopped at epoch {epoch + 1}")

# Instantiate the custom callback with stop button and save functionality
plot_and_stop_callback = PlotLossesAndStop(model_save_path=f'models/my_model_{ver}.h5')


# Step 6: Train the Model
# history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_test, Y_test))
# history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))


# Train the model with the custom callback
history = model.fit(
    X_train, Y_train,
    epochs=25, 
    # batch_size=32, 
    batch_size=32, 
    validation_data=(X_test, Y_test),
    callbacks=[plot_and_stop_callback]
)


# Step 7: Evaluate the Model
loss, mae = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# # Save the entire model to a file
model.save(f'models/my_model_{ver}.h5')  # Saves in HDF5 format

# # Optional: Plotting the Training History
# import matplotlib.pyplot as plt

# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# plt.plot(history.history['mae'], label='Training MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.show()
