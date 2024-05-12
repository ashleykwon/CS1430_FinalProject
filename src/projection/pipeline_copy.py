import numpy as np
from PIL import Image
from geometry import depth_to_points
import cv2
import torch
from zoe_depth import ZoeDepth
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp 
import pickle

zoe_depth = ZoeDepth(device=("cuda" if torch.cuda.is_available() else "cpu"))

def reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r, new_x, new_y) -> Image.Image:
    # Assume K_l and K_r are the same
    # Assume t_r is a zero vector and R_r is an identity matrix
    # Assume leftCameraFrame and rightCameraFrame have the same shape

    # Initialize dataFor3Dto2D  
    leftCamFrameNP = np.asarray(leftCameraFrame)
    rightCamFrameNP = np.asarray(rightCameraFrame)

    dataFor3Dto2D = np.zeros(leftCamFrameNP.shape)

    # TODO: can we do this in one pass?
    leftCameraDepth = zoe_depth.get_depth(leftCameraFrame)
    rightCameraDepth = zoe_depth.get_depth(rightCameraFrame)

    leftCameraTo3D = depth_to_points(
        leftCameraDepth,
        K_l, R_l, t_l
    )
    rightCameraTo3D = depth_to_points(
        rightCameraDepth, 
        K_r, R_r, t_r
    )

    # print(leftCameraTo3D.shape)
    # print(rightCameraTo3D.shape)

    # Sanity check to see if we can project leftCameraTo3D and rightCameraTo3D back to 2D images. Comment this out when actually running the code
    # leftCamExtrinsicMatrix = np.hstack((R_l, t_l))
    # rightCamExtrinsicMatrix = np.hstack((R_r, t_r))
    
    # TODO 3: Do the 3D to 2D mapping + viewing angle modification based on face detection and save the result in dataFor3Dto2D

    # Get the 2D to 3D mapping information using how points in leftCameraFrame get mapped to leftCameraTo3D AND the RGB values of those points
    w, h, c = leftCamFrameNP.shape
    leftCamFrameFlat = leftCamFrameNP.reshape(w*h, c) # RGB values from the left camera frame
    rightCamFrameFlat = rightCamFrameNP.reshape(w*h, c) # RGB values from the right camera frame

    # Derive a new extrinsic matrix for the third camera (user's head) with new_x and new_y
    # newRotationVec = new_x*cv2.Rodrigues(R_l) # check this 
    R_l = R.from_matrix(R_l).as_quat() # Change this to a Rotation instance
    R_r = R.from_matrix(R_r).as_quat() # Change this to a Rotation instance
    slerp = Slerp([0,1], [R_r, R_l])
    newRotationVec = slerp([new_x])[0]
    newRotationVec = cv2.Rodrigues(newRotationVec.as_matrix())

    newTranslationVec = np.multiply(np.asarray([new_x, new_y, 1]), t_l) 

    # Use cv2.projectPoints to derive dataFor3Dto2D (3D points mapped to a 2D image)
    remapped2DCoordsLeft = cv2.projectPoints(leftCameraTo3D, newRotationVec, newTranslationVec, K_l, dist_l) 
    remapped2DCoordsLeft[remapped2DCoordsLeft < 0] = -1
    remapped2DCoordsRight = cv2.projectPoints(rightCameraTo3D, newRotationVec, newTranslationVec, K_l, dist_r) 
    remapped2DCoordsRight[remapped2DCoordsRight < 0] = -1

    for i in range(remapped2DCoordsLeft.shape[0]): # Change this so that it doesn't use for loop
        coordLeft = remapped2DCoordsLeft[i]
        colorLeft = leftCamFrameFlat[i]
        dataFor3Dto2D[coordLeft[0], coordLeft[1], :] = colorLeft
        
        coordRight = remapped2DCoordsRight[i]
        colorRight = rightCamFrameFlat[i]
        dataFor3Dto2D[coordRight[0], coordRight[1], :] = colorRight

    return dataFor3Dto2D



if __name__ == "__main__":
    # K_l = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) 
    # R_l = np.eye(3) 
    # t_l = np.asarray([10, 0, 0])
    # K_r = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    # R_r = np.eye(3) 
    # t_r = np.zeros(3)
    K_l, R_l, t_l, dist_l = pickle.load(open("/Users/ashleykwon/Desktop/CS1430_FinalProject/src/projection/test3Dto2D/left_camera.pickle", 'rb'))
    t_l = np.asarray([t_l[0][0], t_l[1][0], t_l[2][0]])
    K_r, R_r, t_r, dist_r = pickle.load(open("/Users/ashleykwon/Desktop/CS1430_FinalProject/src/projection/test3Dto2D/right_camera.pickle", 'rb'))
    leftCameraFrame = Image.open("/Users/ashleykwon/Desktop/CS1430_FinalProject/src/projection/test3Dto2D/left.jpg")
    # leftCameraFrame = np.asarray(leftCameraFrame)
    rightCameraFrame = Image.open("/Users/ashleykwon/Desktop/CS1430_FinalProject/src/projection/test3Dto2D/right.jpg")
    # rightCameraFrame = np.asarray(rightCameraFrame)
    reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r, 0.5, 0.5)
