import io
import json
import numpy as np
import cv2
import os

from tqdm import tqdm


def get_intrinsic_matrix(fov_x, fov_y, W, H):
    f_x = (W / 2) / np.tan(np.radians(fov_x / 2))
    f_y = (H / 2) / np.tan(np.radians(fov_y / 2))
    s = 0.0
    c_x = W / 2
    c_y = H / 2
    return np.array([[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1]])


def stereo_calibration(K_l, K_r, chessboard_images, chess_box_size_meters):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((10 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[:7, :10].T.reshape(-1, 2)
    objp = objp * chess_box_size_meters

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in left image plane.
    imgpointsR = []  # 2d points in right image plane.

    for imgL, imgR in tqdm(chessboard_images):
        outputL = imgL.copy()
        outputR = imgR.copy()
        retL, cornersL = cv2.findChessboardCorners(image=outputL, patternSize=(7, 10))
        retR, cornersR = cv2.findChessboardCorners(image=outputR, patternSize=(7, 10))

        if retL and retR:
            objpoints.append(objp)

            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            cornersL = cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)

            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            cornersR = cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)

    dist_l = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpointsL,
        imageSize=chessboard_images[0][0].shape[:2][::-1],
        cameraMatrix=None,
        distCoeffs=None
    )[2]

    dist_r = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpointsR,
        imageSize=chessboard_images[0][1].shape[:2][::-1],
        cameraMatrix=None,
        distCoeffs=None,
    )[2]

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Calculate the rotation and translation between the two cameras
    R_l = np.identity(3)
    t_l = np.zeros(3)

    R_r, t_r = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpointsL,
        imagePoints2=imgpointsR,
        cameraMatrix1=K_l,
        distCoeffs1=dist_l,
        cameraMatrix2=K_r,
        distCoeffs2=dist_r,
        imageSize=chessboard_images[0][0].shape[:2][::-1],
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria_stereo,
    )[5:7]

    return K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r
