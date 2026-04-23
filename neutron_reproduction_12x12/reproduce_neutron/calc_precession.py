import numpy as np
import quaternion as quat # https://pypi.org/project/numpy-quaternion/
import sys

from reconstruction_v2.plot_tools import *


def calc_quaternion(Bv, lam, L, voxel_size):

    """ Calc quaternion representation of axis angle rotation
        Maybe look in to doing https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        instead?
    """
    # print(Bv)
    # print(f'Bv: {Bv}')
    B = np.linalg.norm(Bv)
    # print(f'B: {B}')
    c = 4.632e14
    # lam = 2e-10
    # L = 1e-4
    theta = c * lam * B * L * voxel_size
    # print(f'theta: {theta}')
    rot_axis = np.array([0.] + Bv)
    # print(f'rot axis: {rot_axis}')
    if B != 0.0:
        axis_angle = (theta*0.5) * rot_axis/B
    else:
        axis_angle = 0 * rot_axis
    # print(axis_angle)
    qlog = quat.quaternion(*axis_angle)
    q = np.exp(qlog)
    return q


def rotate_vec(v, q=quat.quaternion(0,0,0,0)):

    """rotate vector, v, by quaternion q"""

    vec = np.array([0.] + v)
    vec = quat.quaternion(*v)

    v_prime = q * vec * np.conjugate(q)

    return v_prime.imag

def yrot_quat(theta):

    "calculate quaternion for a rotation around the y axis by angle theta"

    q = quat.quaternion(np.cos(theta*0.5),0,np.sin(theta*0.5),0)
    return q



def calc_precession(nNeutrons, nAngles, angles, wavelengths, voxel_index, voxel_data, voxel_size, B):

    # for holding the "mesurement" result. The last two dimensions are for direction of polarisation and analysis.
    #measurement = np.empty((nNeutrons, nAngles, 3, 3), dtype=object)

    measurement = np.empty((nNeutrons, nAngles, len(wavelengths), 3, 3), dtype=float)

    # starting quaternion
    qs=quat.quaternion(*np.array([1.,0.,0.,0.]))

    # we should loop over starting polarisation along x, y or z
    p_start = [[1,0,0], [0,1,0], [0,0,1]]
    # p_start = np.array([1.0,0.0,0.0])
    for ind_p,p in enumerate(p_start):
    
        # this next loops over each ray and gets vectors for its voxel intersect indices and distance within each voxel.
        # the loop start with first ray at first angle. E.g.:
        # L: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        # v: [ 11  23  35  47  59  71  83  95 107 119 131 143]
        # the  goes to second ray at first angle. E.g.:
        # L: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        # v: [ 10  22  34  46  58  70  82  94 106 118 130 142]
        # and so on.
        n = 0
        a = 0

        # end result vector
        r = p
        for a in range(0,nAngles):

            # we need to rotate B field in each voxel 
            # should probably use something like np.array(list(map(rotate_func, B.reshape((nAngles*nNeutrons?,))))).reshape((nAngles, nNeutrons, 3))   
            # or we dont even need the B field as a matrix, but just a list of x,y,z components?
            theta = angles[a]
            # B2 = np.apply_along_axis(rotate_vec, 2, B, q=yrot_quat(theta))
            Br = np.apply_along_axis(rotate_vec, 1, B, q=yrot_quat(-theta))

            for n in range(0,nNeutrons):

                for ind_l,lam in enumerate(wavelengths):
                    # for vs, Ls in np.nditer([voxel_index, voxel_data], flags=["refs_ok"], order='F'):
                    vs = voxel_index[a][n]
                    Ls = voxel_data[a][n]
                    # We need to rotate the field in the voxels to match the rotation of the sample (the angle of the rays)
                    # maybe just reshape B to be of size [imsize*imsize,3] and then do the rotation for each angle
                    # easier to do in advance. we should do loops over ray and angle seperately
                    # also incoorporate option to have partial dataset??
                    q=qs
                    # for v in np.nditer(vs.flat(), flags=["refs_ok"], order='F'):
                    for ind,v in enumerate(list(vs.flat)):
                        # print(f'v in loop: {v}')
                        ## B = np.array([1.,2.,3.])*1e-3
                        # print(f'Br shape:  {Br.shape}')
                        Bs = Br[v]
                        # print(f'Bs shape:  {Bs.shape}')

                        q = q*calc_quaternion(Bs,lam,Ls[ind],voxel_size)

                        # print(q)
                    # print(p_start)
                    p_end = rotate_vec(p, q=q)
                    # print(p_end)
                    measurement[n][a][ind_l][ind_p] = p_end
   
    return measurement
                #np.empty((nNeutrons, nAngles, 3, 3), dtype=object)
             

if __name__ == "__main__":
   
    path_ver = 1
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


    # just a single wavelength for now
    wavelengths = [4e-10]
    voxel_size = 1e-3
    # generate some random B field in a im_size*im_size grid
    # B = np.random.rand(im_size, im_size, 3)*1e-3
    B = np.random.rand(im_size, im_size, 3)*1e-6
    # B[3,3]=1e-2

    B = B.reshape(im_size*im_size,3, order='F')

    measurement = calc_precession(nNeutrons, nAngles, angles, wavelengths, voxel_index, voxel_data, voxel_size, B)
    print('*'*20)
    print(measurement[:, :, 0, 0].shape)
    # ms = measurement[:, :, 0, 0]
    # ms = ms.flatten()
    # ms = np.reshape(ms, (16,10))
    # print('*'*20)
    # plt.imshow(ms, interpolation='none', vmin=-1, vmax=1)
    # plt.show()  

    fig, axs = plt.subplots(3, 3)
    for sp in range(0,3):
        for sa in range(0,3):
                im = axs[sa, sp].imshow(measurement[:, :, 0, sp, sa], interpolation='none', cmap='viridis')
                plt.colorbar(im, ax=axs[sa, sp], fraction=0.046, pad=0.04)
    
    plt.show()
