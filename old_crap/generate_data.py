import numpy as np
from calc_precession import calc_precession

   
global voxel_index
global voxel_data
global wavelengths
global nNeutrons
global nAngles
global angles

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


num_samples = 20
# just a single wavelength for now
# wavelengths = [2e-10, 2.5e-10, 3e-10, 3.5e-10, 4e-10, 4.5e-10, 5e-10, 5.5e-10, 6e-10, 6.5e-10, 7e-10, 7.5e-10, 8e-10]
wavelengths = np.arange(2,8,0.2)*1e-10
voxel_size = 1e-3
# generate some random B field in a im_size*im_size grid
# B = np.random.rand(im_size, im_size, 3)*1e-3
seed = 0
np.random.seed(seed)

B_data = np.empty((num_samples, nNeutrons, nAngles, len(wavelengths), 3, 3), dtype=float)
A_data = np.empty((num_samples, im_size, im_size, 3), dtype=float)

min_B = -5e-3
max_B = +5e-3

for i in range(0,num_samples):
        print(f'generating sample set {i} out of {num_samples}')
        # B = np.zeros((im_size, im_size, 3), dtype=float)
        # # B[1,1,:]=np.random.rand(3)*1e-3
        # # B[1,2,:]=np.random.rand(3)*1e-3
        # # B[2,1,:]=np.random.rand(3)*1e-3
        # # B[2,2,:]=np.random.rand(3)*1e-3
        # B[1:3,1:3,:] = min_B + (max_B - min_B) * np.random.rand(2, 2, 3)
        # #print(B)

        B = min_B + (max_B - min_B) * np.random.rand(im_size, im_size, 3)
        A_data[i,:,:,:] = B
        

        B = B.reshape(im_size*im_size,3, order='F')


        # B_data[i,:,:,:,:,:] = calc_precession(B)
        B_data[i,:,:,:,:,:] = calc_precession(nNeutrons, nAngles, angles, wavelengths, voxel_index, voxel_data, voxel_size, B)



np.save(f'data/A_data_{ver}.npy', A_data, allow_pickle=True, fix_imports=False)
np.save(f'data/B_data_{ver}.npy', B_data, allow_pickle=True, fix_imports=False)
np.savez(f'data/meta_{ver}.npz', seed=seed, wavelengths=wavelengths, num_samples=num_samples, voxel_size=voxel_size)