import numpy as np

def get_intrinsic_matrix(fov_x, fov_y, W, H):
    f_x = (W / 2) / np.tan(np.radians(fov_x / 2))
    f_y = (W / 2) / np.tan(np.radians(fov_y / 2))
    s = 0.0
    c_x = W / 2
    c_y = H / 2
    return np.array([
        [f_x, s, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

if __name__ == '__main__':
    print(get_intrinsic_matrix(fov_x=82.1, fov_y=52.2, W=1920, H=1080))
