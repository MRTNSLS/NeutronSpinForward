import sys
import numpy as np
from calc_precession import calc_precession
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
from interactive_data_plot import interactive_plot
global voxel_index
global voxel_data
global wavelengths
global nNeutrons
global nAngles
global angles

ver = 105

model = load_model(f'models/my_model_{ver}.h5')

# Summarize the model architecture
model.summary()


with np.load(f'models/scalers_{ver}.npz') as data:
        X_min = data['X_min']
        Y_min = data['Y_min']
        X_max = data['X_max']
        Y_max = data['Y_max']
        
with np.load(f'data/meta_{ver}.npz', allow_pickle=True) as data:
        wavelengths = data['wavelengths']
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
        num_samples = data['num_samples']
        wavelengths = data['wavelengths']
        voxel_size = data['voxel_size']
        minmax_B = data['minmax_B']




# generate some random B field in a im_size*im_size grid
# B = np.random.rand(im_size, im_size, 3)*1e-3
seed = 6
np.random.seed(seed)


# B = np.zeros((im_size, im_size, 3), dtype=float)
# B[1:3,1:3,:] = min_B + (max_B - min_B) * np.random.rand(2, 2, 3)
# B = min_B + (max_B - min_B) * np.random.rand(im_size, im_size, 3)
B = np.zeros((im_size, im_size, 3))
B[1:-1,1:-1,:] = -minmax_B + (minmax_B + minmax_B) * np.random.rand(im_size-2, im_size-2, 3)
# B = -minmax_B + (minmax_B + minmax_B) * np.random.rand(im_size, im_size, 3)

B2 = B.reshape(im_size*im_size,3, order='F')

D = np.empty((1, nNeutrons, nAngles, len(wavelengths), 3, 3), dtype=float)
D0 = calc_precession(nNeutrons, nAngles, angles, wavelengths, voxel_index, voxel_data, voxel_size, B2)
D[0,:,:,:,:,:] = D0
interactive_plot(D0)



# New data for predictions
# X_flattened = D.reshape(1, -1)
# X_new_scaled = scaler_X.transform(X_flattened)  # Normalize new input data
# X_new_scaled = X_new_scaled.reshape(D.shape)

X_new_scaled = (D - X_min) / (X_max - X_min)
# Get predictions (still in normalized form)
predictions_scaled = model.predict(X_new_scaled)
# predictions_flattened = predictions_scaled.reshape(1, -1)
# Unnormalize the predictions using the output scaler
# predictions_original = scaler_Y.inverse_transform(predictions_flattened)
# predictions_original = predictions_original.reshape(predictions_scaled.shape)
# print(predictions_original)
# print(predictions_original.shape)
predictions_original = predictions_scaled * (Y_max - Y_min) + Y_min

fig, axs = plt.subplots(3, 3)

scale = 1e3

for i in range(0, 3):
    i1 = B[:, :, i] * scale
    i2 = predictions_original[0, :, :, i] * scale
    i3 = (i1 - i2)

    # Independent scaling for Truth and Recon
    vmax_true = np.abs(i1).max() * 1.1
    if vmax_true < 1e-9: vmax_true = 1e-9
    vmax_pred = np.abs(i2).max() * 1.1
    if vmax_pred < 1e-9: vmax_pred = 1e-9
    
    p1 = axs[0, i].imshow(i1, interpolation='none', vmin=-vmax_true, vmax=vmax_true, cmap='coolwarm')
    fig.colorbar(p1, ax=axs[0, i])
    axs[0, i].set_title(f"True {'XYZ'[i]}")

    p2 = axs[1, i].imshow(i2, interpolation='none', vmin=-vmax_pred, vmax=vmax_pred, cmap='coolwarm')
    fig.colorbar(p2, ax=axs[1, i])
    axs[1, i].set_title(f"Pred {'XYZ'[i]}")

    vmax_err = np.abs(i3).max() * 1.1
    if vmax_err < 1e-9: vmax_err = 1e-9
    p3 = axs[2, i].imshow(i3, interpolation='none', vmin=-vmax_err, vmax=vmax_err, cmap='RdBu_r')
    fig.colorbar(p3, ax=axs[2, i])
    axs[2, i].set_title(f"Error {'XYZ'[i]}")

    
plt.show() 