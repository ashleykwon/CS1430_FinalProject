import io
import json
import numpy as np
import cv2
import os

from tqdm import tqdm


def get_intrinsic_matrix(fov_x, fov_y, W, H):
    f_x = (W / 2) / np.tan(np.radians(fov_x / 2))
    f_y = (W / 2) / np.tan(np.radians(fov_y / 2))
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
        retL, cornersL = cv2.findChessboardCorners(image=imgL, patternSize=(7, 10))
        retR, cornersR = cv2.findChessboardCorners(image=imgR, patternSize=(7, 10))

        if retL and retR:
            objpoints.append(objp)

            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            cornersL = cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)

            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            cornersR = cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)

    K_l_, distL = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpointsL,
        imageSize=chessboard_images[0][0].shape[:2][::-1],
        # cameraMatrix=None,
        # distCoeffs=None,
        cameraMatrix=K_l.copy(),
        distCoeffs=None,
        flags=(cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT)
    )[1:3]

    K_r_, distR = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpointsR,
        imageSize=chessboard_images[0][1].shape[:2][::-1],
        cameraMatrix=None,
        distCoeffs=None,
        # cameraMatrix=K_r,
        # distCoeffs=None,
        # flags=(cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT)
    )[1:3]

    print("K_l")
    print(K_l)
    print("K_l_")
    print(K_l_)

    print("K_r")
    print(K_r)
    print("K_r_")
    print(K_r_)

    print("distL")
    print(distL)
    print("distR")
    print(distR)

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
    retS, mtxL, distL, mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
        objectPoints=objpoints,
        imagePoints1=imgpointsL,
        imagePoints2=imgpointsR,
        cameraMatrix1=K_l,
        distCoeffs1=distL,
        cameraMatrix2=K_r,
        distCoeffs2=distR,
        imageSize=chessboard_images[0][0].shape[:2],
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria_stereo,
    )

    R_l = Rot
    t_l = Trns
    R_r = np.identity(3)
    t_r = np.zeros(3)

    print("R_l")
    print(R_l)
    print("t_l")
    print(t_l)
    print("R_r")
    print(R_r)
    print("t_r")
    print(t_r)

    return R_l, t_l, R_r, t_r
