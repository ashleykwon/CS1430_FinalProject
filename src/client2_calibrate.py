import os
import numpy as np
from projection.camera import get_intrinsic_matrix, stereo_calibration
import tyro
import cv2
from tqdm import tqdm
import time
import pickle


def collect_images(N: int =10):
    # if not os.path.exists("1"):
    #     os.makedirs("1")
    # if not os.path.exists("2"):
    #     os.makedirs("2")
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    chessboard_images = []
    for i in tqdm(range(N)):
        time.sleep(2)
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        chessboard_images.append([frame1, frame2])
        # cv2.imwrite("1/" + str(i) + ".jpg", frame1)
        # cv2.imwrite("2/" + str(i) + ".jpg", frame2)
    return chessboard_images

def main(
    N: int = 20,
    chess_box_size_meters: float = 0.0235,
    fov_x: float = 82.1,
    fov_y: float = 52.2,
    W: int = 1920,
    H: int = 1080,
    left_camera_output_file: str = 'left_camera.pickle',
    right_camera_output_file: str = 'right_camera.pickle',
):
    chessboard_images = collect_images(N)
    K_l = get_intrinsic_matrix(fov_x=fov_x, fov_y=fov_y, W=W, H=H)
    K_r = get_intrinsic_matrix(fov_x=fov_x, fov_y=fov_y, W=W, H=H)
    R_l, t_l, R_r, t_r = stereo_calibration(K_l, K_r, chessboard_images, chess_box_size_meters)
    pickle.dump((K_l, R_l, t_l), open(left_camera_output_file, "wb"))
    pickle.dump((K_r, R_r, t_r), open(right_camera_output_file, "wb"))

if __name__ == '__main__':
    tyro.cli(main)
