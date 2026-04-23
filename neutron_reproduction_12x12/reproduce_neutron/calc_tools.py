try:
    from . import ray_wrapper
except ImportError:
    import ray_wrapper
import numpy as np

def sou_det_calc(nNeutrons, Bnsizxrecon, angles, scaleD):
    Dfac = 1.0001
    D = Dfac * Bnsizxrecon
    h = D / nNeutrons

    detector = scaleD * D * np.ones(nNeutrons) + 1j * h * np.linspace(-nNeutrons / 2 + 0.5, nNeutrons / 2 - 0.5, nNeutrons)
    source = -scaleD * D * np.ones(nNeutrons) + 1j * h * np.linspace(-nNeutrons / 2 + 0.5, nNeutrons / 2 - 0.5, nNeutrons)

    sou_x = np.imag(source[:, np.newaxis] * np.exp(1j * angles))
    sou_x = -sou_x.flatten()

    sou_z = np.real(source[:, np.newaxis] * np.exp(1j * angles))
    sou_z = sou_z.flatten()

    det_x = np.imag(detector[:, np.newaxis] * np.exp(1j * angles))
    det_x = -det_x.flatten()

    det_z = np.real(detector[:, np.newaxis] * np.exp(1j * angles))
    det_z = det_z.flatten()

    return sou_x, sou_z, det_x, det_z


if __name__ == "__main__":
    # Define image size (grid dimensions)
    im_size = 12

    nNeutrons = 3
    nAngles = 1
    angles = np.linspace(0, np.pi*2, nAngles, endpoint=False)  # Assuming angles in radians
    scaleD = 2

    source_xs, source_zs, det_xs, det_zs = sou_det_calc(nNeutrons, im_size, angles, scaleD)
    print(source_xs)
    print(source_zs)
    print(det_xs)
    print(det_zs)