import io
import json
import numpy as np
import cv2 as cv
import os

from tqdm import tqdm


def get_intrinsic_matrix(fov_x, fov_y, W, H):
    f_x = (W / 2) / np.tan(np.radians(fov_x / 2))
    f_y = (W / 2) / np.tan(np.radians(fov_y / 2))
    s = 0.0
    c_x = W / 2
    c_y = H / 2
    return np.array([[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1]])


def stereo_calibration(K_l, K_r, chessboard_images, chess_box_size_mm):
    if not os.path.exists("1_corners"):
        os.makedirs("1_corners")
    if not os.path.exists("2_corners"):
        os.makedirs("2_corners")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:10].T.reshape(-1, 2)
    objp = objp * chess_box_size_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in left image plane.
    imgpointsR = []  # 2d points in right image plane.

    for i in tqdm(range(len(chessboard_images))):
        imgL = cv.imread("1/%d.jpg" % i)
        imgR = cv.imread("2/%d.jpg" % i)
        imgL_gray = cv.imread("1/%d.jpg" % i, 0)
        imgR_gray = cv.imread("2/%d.jpg" % i, 0)

        outputL = imgL.copy()
        outputR = imgR.copy()

        retR, cornersR = cv.findChessboardCorners(outputR, (7, 10), None)
        retL, cornersL = cv.findChessboardCorners(outputL, (7, 10), None)

        if retR and retL:
            objpoints.append(objp)
            cv.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
            cv.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(outputR, (7, 10), cornersR, retR)
            cv.drawChessboardCorners(outputL, (7, 10), cornersL, retL)
            cv.imwrite("1_corners/%d.jpg" % i, outputL)
            cv.imwrite("2_corners/%d.jpg" % i, outputR)

            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

    # Calibrating left camera
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objpoints, imgpointsL, imgL_gray.shape[::-1], None, None
    )

    # Calibrating right camera
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objpoints, imgpointsR, imgR_gray.shape[::-1], None, None
    )

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamental matrix
    retS, mtxL, distL, mtxR, distR, Rot, Trns, Emat, Fmat = cv.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K_l,
        distL,
        K_r,
        distR,
        imgL_gray.shape[::-1],
        criteria_stereo,
        flags,
    )
    
    R_l = Rot
    t_l = Trns
    R_r = np.identity(3)
    t_r = np.zeros(3)
    
    return R_l, t_l, R_r, t_r
