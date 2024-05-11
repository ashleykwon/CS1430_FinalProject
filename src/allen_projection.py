import io
import json
import os
import time
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tqdm import tqdm

# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config


# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# https://gist.github.com/hprobotic/0dc912f69483c3bdf578d4315249820a


# Should have Cam 1 as VideoCapture(0), as Left Frame, connected to Left USB
# Should have Cam 2 as VideoCapture(1), as Right Frame, connected to Right USB
cams = {"1": 0, "2": 1}


# 1 Find camera calibration matrices (instrinsic, extrinsic) one time thing
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
# https://nikatsanka.github.io/camera-calibration-using-opencv-and-python.html


def take_calibration_images():
    for name in cams:
        if not os.path.exists(name):
            os.makedirs(name)
    cap1 = cv.VideoCapture(0)
    cap2 = cv.VideoCapture(1)
    for i in tqdm(range(10)):
        time.sleep(2)
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        cv.imwrite("1/" + str(i) + ".jpg", frame1)
        cv.imwrite("2/" + str(i) + ".jpg", frame2)


# https://github.com/IntelRealSense/librealsense/blob/master/doc/depth-from-stereo.md
# https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/
def calibrate_cameras():
    for name in cams:
        if not os.path.exists(name + "_corners"):
            os.makedirs(name + "_corners")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 10, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:10].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in left image plane.
    imgpointsR = []  # 2d points in right image plane.

    for i in tqdm(range(10)):
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
    mtxL = np.array([[1102.40922, 0.0, 960], [0.0, 1102.40922, 540], [0.0, 0.0, 1.0]])
    camLInt = np.asarray(mtxL).tolist()
    # hL,wL= imgL_gray.shape[:2]
    # new_mtxL, roiL= cv.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

    # Calibrating right camera
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objpoints, imgpointsR, imgR_gray.shape[::-1], None, None
    )
    mtxR = np.array([[1102.40922, 0.0, 960], [0.0, 1102.40922, 540], [0.0, 0.0, 1.0]])
    camRInt = np.asarray(mtxR).tolist()
    # hR,wR= imgR_gray.shape[:2]
    # new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,(wR,hR),1,(wR,hR))

    # Write JSON file
    with io.open("camLInt.json", "w", encoding="utf8") as outfile:
        str_ = json.dumps(camLInt, indent=4, sort_keys=True, separators=(",", ": "))
        outfile.write(str_)

    with io.open("camRInt.json", "w", encoding="utf8") as outfile:
        str_ = json.dumps(camRInt, indent=4, sort_keys=True, separators=(",", ": "))
        outfile.write(str_)

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
        mtxL,
        distL,
        mtxR,
        distR,
        imgL_gray.shape[::-1],
        criteria_stereo,
        flags,
    )
    camLExt = {"Rot": np.asarray(Rot).tolist(), "Trns": np.asarray(Trns).tolist()}
    camRExt = {
        "Rot": np.asarray(np.identity(3)).tolist(),
        "Trns": np.asarray(np.zeros(3)).tolist(),
    }
    with io.open("camLExt.json", "w", encoding="utf8") as outfile:
        str_ = json.dumps(camLExt, indent=4, sort_keys=True, separators=(",", ": "))
        outfile.write(str_)
    with io.open("camRExt.json", "w", encoding="utf8") as outfile:
        str_ = json.dumps(camRExt, indent=4, sort_keys=True, separators=(",", ": "))
        outfile.write(str_)

    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv.stereoRectify(
        mtxL,
        distL,
        mtxR,
        distR,
        imgL_gray.shape[::-1],
        Rot,
        Trns,
        rectify_scale,
        (0, 0),
    )

    with io.open("Q.json", "w", encoding="utf8") as outfile:
        str_ = json.dumps(
            np.asarray(Q).tolist(), indent=4, sort_keys=True, separators=(",", ": ")
        )
        outfile.write(str_)


def get_calibrations():
    # Opening JSON file
    with open("camLInt.json", "r") as openfile:
        # Reading from json file
        camLInt = json.load(openfile)
        camLInt = np.asarray(camLInt)
    with open("camRInt.json", "r") as openfile:
        # Reading from json file
        camRInt = json.load(openfile)
        camRInt = np.asarray(camRInt)
    with open("camLExt.json", "r") as openfile:
        # Reading from json file
        camLExt = json.load(openfile)
        camLExt = {
            "Rot": np.asarray(camLExt["Rot"]),
            "Trns": np.asarray(camLExt["Trns"]),
        }
    with open("camRExt.json", "r") as openfile:
        # Reading from json file
        camRExt = json.load(openfile)
        camRExt = {
            "Rot": np.asarray(camRExt["Rot"]),
            "Trns": np.asarray(camRExt["Trns"]),
        }
    with open("Q.json", "r") as openfile:
        # Reading from json file
        Q = json.load(openfile)
        Q = np.asarray(Q)
    return camLInt, camRInt, camLExt, camRExt, Q


# 2 Find SIFT features b/w images and matches
def take_stereo_images():
    cap1 = cv.VideoCapture(0)
    cap2 = cv.VideoCapture(1)
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    cv.imwrite("stereo_1.jpg", frame1)
    cv.imwrite("stereo_2.jpg", frame2)


# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def features_and_matching():
    img_1 = cv.imread("stereo_1.jpg", cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread("stereo_2.jpg", cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(img_1, None)
    kp_2, des_2 = sift.detectAndCompute(img_2, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_1, des_2, k=2)

    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp_2[m.trainIdx].pt)
            pts1.append(kp_1[m.queryIdx].pt)

    return img_1, img_2, pts1, pts2


# 3 Fundamental mats and then find epipolar lines
# https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img_1, img_2, lines, pts_1, pts_2):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""
    r, c = img_1.shape
    img_1 = cv.cvtColor(img_1, cv.COLOR_GRAY2BGR)
    img_2 = cv.cvtColor(img_2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts_1, pts_2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img_1 = cv.line(img_1, (x0, y0), (x1, y1), color, 1)
        img_1 = cv.circle(img_1, tuple(pt1), 5, color, -1)
        img_2 = cv.circle(img_2, tuple(pt2), 5, color, -1)
    return img_1, img_2


def images_with_epipolars(img_1, img_2, pts_1, pts_2):
    pts_1 = np.int32(pts_1)
    pts_2 = np.int32(pts_2)
    F, mask = cv.findFundamentalMat(pts_1, pts_2, cv.FM_LMEDS)

    # We select only inlier points
    pts_1 = pts_1[mask.ravel() == 1]
    pts_2 = pts_2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts_2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img_L, _ = drawlines(img_1, img_2, lines1, pts_1, pts_2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts_1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img_R, _ = drawlines(img_2, img_1, lines2, pts_2, pts_1)

    cv.imwrite("epipolar_1.png", img_L)
    cv.imwrite("epipolar_2.png", img_R)
    return F, pts_1, pts_2, img_L, img_R


# 4 Rectify images based on 3
# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
# https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system
def stereo_rectification(F, pts_1, pts_2, img_1, img_2):
    h1, w1, _ = img_1.shape
    h2, w2, _ = img_2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts_1), np.float32(pts_2), F, imgSize=(w1, h1)
    )
    img1_rectified = cv.warpPerspective(img_1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img_2, H2, (w2, h2))
    cv.imwrite("rectified_1.png", img1_rectified)
    cv.imwrite("rectified_2.png", img2_rectified)
    return img1_rectified, img2_rectified


# 5 Dense correspondences from 4
# 6 Triangulation based on dense correspondences = depth
def disparity(img_l, img_r):
    grayLeft = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
    grayRight = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

    stereo = cv.StereoBM.create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(grayLeft, grayRight)
    # plt.imshow(disparity,'gray')
    # plt.show()
    return disparity


# 7 Proj (2D image, depth) => 3D point cloud
# 8 3D => 2D proj. using pseudo camera with derived extrinsics


if __name__ == "__main__":
    # take_calibration_images()
    # calibrate_cameras()
    camLInt, camRInt, camLExt, camRExt, Q = get_calibrations()

    # take_stereo_images()
    img_1, img_2, pts_1, pts_2 = features_and_matching()
    if len(pts_1) < 9:
        print("rip")
    F, pts_1, pts_2, img_L, img_R = images_with_epipolars(img_1, img_2, pts_1, pts_2)
    img1_rect, img2_rect = stereo_rectification(F, pts_1, pts_2, img_L, img_R)
    disp = disparity(img1_rect, img2_rect)
    depth = cv.reprojectImageTo3D(disp, Q)
    print(depth)
    plt.imshow(depth)
    plt.show(depth)

    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_zoe_n = torch.hub.load("./ZoeDepth", "ZoeD_N", source="local", pretrained=True)
    # conf = get_config("zoedepth", "infer")
    # model_zoe_n = build_model(conf)
