import numpy as np
from calc_precession import calc_precession
from matplotlib.widgets import Slider, Button

try:
    from calc_tools import sou_det_calc, ray_wrapper
except ImportError:
    import ray_wrapper
    from calc_tools import sou_det_calc

# Global parameters
ver = 106
im_size = 6
nNeutrons = 31
nAngles = 91
angles = np.linspace(0, np.pi*2, nAngles, endpoint=False)  # angles in radians
wavelengths = np.arange(2,8,0.2)*1e-10
sample_size = 1e-2
voxel_size = sample_size/im_size

def generate_data():
    scaleD = 2
    num_samples = 1000
    seed = 1
    np.random.seed(seed)
    minmax_B = 5e-3

    source_xs, source_zs, det_xs, det_zs = sou_det_calc(nNeutrons, im_size, angles, scaleD)
    source_xs = np.reshape(source_xs, (nAngles, nNeutrons), order='F')
    source_zs = np.reshape(source_zs, (nAngles, nNeutrons), order='F')
    det_xs = np.reshape(det_xs, (nAngles, nNeutrons), order='F')
    det_zs = np.reshape(det_zs, (nAngles, nNeutrons), order='F')

    voxel_index = np.empty((nAngles, nNeutrons), dtype=object)
    voxel_data = np.empty((nAngles, nNeutrons), dtype=object)

    for n in range(0, nNeutrons):
        for a in range(0, nAngles):
            source_x = source_xs[a][n]
            source_z = source_zs[a][n]
            det_x = det_xs[a][n]
            det_z = det_zs[a][n]

            result = ray_wrapper.ray_wrapper(source_x, source_z, det_x, det_z, im_size)
            voxel_index_t, voxel_data_t = result
            voxel_index[a][n] = voxel_index_t
            voxel_data[a][n] = voxel_data_t

    B_data = np.empty((num_samples, nNeutrons, nAngles, len(wavelengths), 3, 3), dtype=float)
    A_data = np.empty((num_samples, im_size, im_size, 3), dtype=float)

    for i in range(0, num_samples):
        print(f'generating sample set {i} out of {num_samples}')
        B_img = np.zeros((im_size, im_size, 3))
        B_img[1:-1,1:-1,:] = -minmax_B + 2 * minmax_B * np.random.rand(im_size-2, im_size-2, 3)
        A_data[i,:,:,:] = B_img
        
        B_flat = B_img.reshape(im_size*im_size, 3, order='F')
        B_data[i,:,:,:,:,:] = calc_precession(nNeutrons, nAngles, angles, wavelengths, voxel_index, voxel_data, voxel_size, B_flat)

    np.save(f'data/A_data_{ver}.npy', A_data, allow_pickle=True, fix_imports=False)
    np.save(f'data/B_data_{ver}.npy', B_data, allow_pickle=True, fix_imports=False)
    np.savez(f'data/meta_{ver}.npz', 
             im_size=im_size, 
             voxel_index=voxel_index, 
             voxel_data=voxel_data,
             nNeutrons=nNeutrons,
             nAngles=nAngles,
             angles=angles,
             source_xs=source_xs,
             source_zs=source_zs,
             det_xs=det_xs,
             det_zs=det_zs,
             seed=seed, 
             wavelengths=wavelengths, 
             num_samples=num_samples, 
             voxel_size=voxel_size,
             minmax_B=minmax_B)

if __name__ == '__main__':
    generate_data()