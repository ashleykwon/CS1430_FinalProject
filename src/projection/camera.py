import numpy as np


def get_intrinsic_matrix(fov_x, fov_y, W, H):
    f_x = (W / 2) / np.tan(np.radians(fov_x / 2))
    f_y = (W / 2) / np.tan(np.radians(fov_y / 2))
    s = 0.0
    c_x = W / 2
    c_y = H / 2
    return np.array([[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1]])
