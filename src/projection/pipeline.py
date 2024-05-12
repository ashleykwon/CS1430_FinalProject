import numpy as np
from PIL import Image
from .geometry import depth_to_points
import cv2

def reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, R_l, t_l, K_r, R_r, t_r, new_x, new_y) -> Image.Image:
    dataFor3Dto2D = rightCameraFrame

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
    leftCamExtrinsicMatrix = np.hstack((R_l, t_l))
    rightCamExtrinsicMatrix = np.hstack((R_r, t_r))
    leftCamFrameReconstructed = np.linalg.inv(K_l)@np.linalg.inv(leftCamExtrinsicMatrix)@leftCameraTo3D
    rightCamFrameReconstructed = np.linalg.inv(K_r)@np.linalg.inv(rightCamExtrinsicMatrix)@rightCameraTo3D
    cv2.imwrite('left_frame_reconstructed.png', leftCamFrameReconstructed)
    cv2.imwrite('right_frame_reconstructed.png', rightCamFrameReconstructed)

    # TODO 3: Do the 3D to 2D mapping + viewing angle modification based on face detection and save the result in dataFor3Dto2D

    # Get the 2D to 3D mapping information using how points in leftCameraFrame get mapped to leftCameraTo3D AND the RGB values of those points
    w, h, c = leftCameraFrame.shape
    leftCamFrameFlat = leftCameraFrame.reshape(w*h, c) # RGB values from the left camera frame
    rightCamFrameFlat = rightCameraFrame.reshape(w*h, c) # RGB values from the right camera frame

    # Derive a new extrinsic matrix for the third camera (user's head) with new_x and new_y
    
    # Use cv2.projectPoints to derive dataFor3Dto2D (3D points mapped to a 2D image)

    dataFor3Dto2D = rightCameraFrame  # change this to an actual output
    return dataFor3Dto2D
