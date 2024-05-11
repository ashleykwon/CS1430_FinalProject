import numpy as np
from projection.camera import get_intrinsic_matrix, stereo_calibration
import tyro


def collect_images(N: int =10):
    pass

def main(
    N: int = 10,
    chess_box_size_mm: int = 23,
    fov_x: float = 82.1,
    fov_y: float = 52.2,
    W: int = 1920,
    H: int = 1080,
    left_camera_output_file: str = 'left_camera.npy',
    right_camera_output_file: str = 'right_camera.npy',
):
    chessboard_images = collect_images(N=10)
    K_l = get_intrinsic_matrix(fov_x=fov_x, fov_y=fov_y, W=W, H=H)
    K_r = get_intrinsic_matrix(fov_x=fov_x, fov_y=fov_y, W=W, H=H)
    R_l, t_l, R_r, t_r = stereo_calibration(K_l, K_r, chessboard_images, chess_box_size_mm)
    np.save(left_camera_output_file, (K_l, R_l, t_l))
    np.save(right_camera_output_file, (K_r, R_r, t_r))

if __name__ == '__main__':
    tyro.cli(main)
