import glob
import os
import time
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# # Depth Map from Stereo Images 
# # https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
# cap0 = cv.VideoCapture(0)
# cap1 = cv.VideoCapture(1)

# ret, frame = cap0.read()
# ret1, frame1 = cap1.read()
# w, h, c = frame.shape
# w1, h1, c1 = frame1.shape
# frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
 
# stereo = cv.StereoBM.create(numDisparities=16, blockSize=5)
# disparity = stereo.compute(frame_gray,frame1_gray)
# plt.imshow(disparity,'gray')
# plt.show()

cams = {"1": 0, "2": 1}

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# 1 Find camera calibration matrices (instrinsic, extrinsic) one time thing
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/
# https://nikatsanka.github.io/camera-calibration-using-opencv-and-python.html

def take_calibration_images(cam_name):
    if not os.path.exists(cam_name):
        os.makedirs(cam_name)
    cap = cv.VideoCapture(cams[cam_name])
    for i in range(10):
        ret, frame = cap.read()
        cv.imwrite(cam_name + '/' + str(i) + '.jpg', frame)
        time.sleep(1)

def calibrate_camera(cam_name):
    if not os.path.exists(cam_name + '_corners'):
        os.makedirs(cam_name + '_corners')

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*10,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(str(cam_name) + '/*.jpg')
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,10), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,10), corners2, ret)
        cv.imshow('img', img)
        cv.imwrite(cam_name + '_corners/' + fname, img)
        cv.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret)
    print("Camera matrix:") 
    print(mtx)
    print("\nDistortion coefficient:") 
    print(dist)
    print("\nRotation Vectors:") 
    print(rvecs)
    print("\nTranslation Vectors:") 
    print(tvecs)

# 2 Find SIFT features b/w images and matches
def take_stereo_images():
    cap1 = cv.VideoCapture(0)
    cap2 = cv.VideoCapture(1)
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    cv.imwrite('stereo_1.jpg', frame1)
    cv.imwrite('stereo_2.jpg', frame2)

# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def features_and_matching():
    img_1 = cv.imread('stereo_1.jpg', cv.IMREAD_GRAYSCALE)
    img_2 = cv.imread('stereo_2.jpg', cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(img_1,None)
    kp_2, des_2 = sift.detectAndCompute(img_2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_1,des_2,k=2)
    
    pts1 = []
    pts2 = []
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp_2[m.trainIdx].pt)
            pts1.append(kp_1[m.queryIdx].pt)
    
    return img_1, img_2, pts1, pts2

# 3 Fundamental mats and then find epipolar lines
# https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img_1,img_2,lines,pts_1,pts_2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c = img_1.shape
    img_1 = cv.cvtColor(img_1,cv.COLOR_GRAY2BGR)
    img_2 = cv.cvtColor(img_2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts_1,pts_2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img_1 = cv.line(img_1, (x0,y0), (x1,y1), color,1)
        img_1 = cv.circle(img_1,tuple(pt1),5,color,-1)
        img_2 = cv.circle(img_2,tuple(pt2),5,color,-1)
    return img_1,img_2

def images_with_epipolars(img_1, img_2, pts_1, pts_2):
    pts_1 = np.int32(pts_1)
    pts_2 = np.int32(pts_2)
    F, mask = cv.findFundamentalMat(pts_1,pts_2,cv.FM_LMEDS)
    
    # We select only inlier points
    pts_1 = pts_1[mask.ravel()==1]
    pts_2 = pts_2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts_2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img_5,img_6 = drawlines(img_1,img_2,lines1,pts_1,pts_2)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts_1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img_3,img_4 = drawlines(img_2,img_1,lines2,pts_2,pts_1)
    
    plt.subplot(121),plt.imshow(img_5)
    plt.subplot(122),plt.imshow(img_3)
    plt.show()

# 4 Rectify images based on 4

# 5 Dense correspondences from 4/5
# 6 Triangulation based on dense correspondences = depth
# 7 Proj (2D image, dpeth) => 3D point cloud
# 8 3D => 2D proj. using pseudo camera with derived extrinsics


# Also could be worth trying the transformer model for stereo
# https://github.com/autonomousvision/unimatch

if __name__ == '__main__':
    # take_calibration_images("1")
    # take_calibration_images("2")
    # calibrate_camera("1")
    # calibrate_camera("2")
    # take_stereo_images()
    img_1, img_2, pts_1, pts_2 = features_and_matching()
    images_with_epipolars(img_1, img_2, pts_1, pts_2)
