from calc_tools import *

path_ver = 6
# Define image size (grid dimensions)
im_size = 3

nNeutrons = 15
nAngles = 181
angles = np.linspace(0, np.pi*2, nAngles, endpoint=False)  # angles in radians
scaleD = 2

source_xs, source_zs, det_xs, det_zs = sou_det_calc(nNeutrons, im_size, angles, scaleD)
source_xs = np.reshape(source_xs, (nAngles, nNeutrons), order='F')
source_zs = np.reshape(source_zs, (nAngles, nNeutrons), order='F')
det_xs = np.reshape(det_xs, (nAngles, nNeutrons), order='F')
det_zs = np.reshape(det_zs, (nAngles, nNeutrons), order='F')


voxel_index = np.empty((nAngles, nNeutrons), dtype=object)
voxel_data = np.empty((nAngles, nNeutrons), dtype=object)
# for index, source_x in enumerate(source_xs):
for n in range(0, nNeutrons):
    for a in range(0, nAngles):
        source_x = source_xs[a][n]
        source_z = source_zs[a][n]
        det_x = det_xs[a][n]
        det_z = det_zs[a][n]

        # Call the ray_wrapper function
        result = ray_wrapper.ray_wrapper(source_x, source_z, det_x, det_z, im_size)

        # Unpack the returned tuple
        voxel_index_t, voxel_data_t = result

        voxel_index[a][n] = voxel_index_t
        voxel_data[a][n] = voxel_data_t
        # # Convert NumPy arrays to Python lists for easier manipulation
        # voxel_index.append(voxel_index_t.tolist())
        # voxel_data.append(voxel_data_t.tolist())

print(voxel_data)

np.savez(f'data/path_data_{path_ver}.npz', 
         im_size=im_size, 
         voxel_index=voxel_index, 
         voxel_data=voxel_data,
         nNeutrons=nNeutrons,
         nAngles=nAngles,
         angles=angles,
         source_xs=source_xs,
         source_zs=source_zs,
         det_xs=det_xs,
         det_zs=det_zs)