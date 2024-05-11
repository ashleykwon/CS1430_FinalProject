import numpy as np
from PIL import Image
from .geometry import depth_to_points

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
        rightCameraDepth, K_r, R_r, t_r
    )

    # print(leftCameraTo3D.shape)
    # print(rightCameraTo3D.shape)

    # TODO 3: Do the 3D to 2D mapping + viewing angle modification based on face detection and save the result in dataFor3Dto2D
    # dataFor3Dto2D = b'sample output' # Change this to the actual output to client 1
    # dataFor3Dto2D SHOULD BE A NUMPY ARRAY

    # translation = np.multiply(leftCameraTranslation, np.asarray([faceCoordinate[0], faceCoordinate[1], 0]))
    # rotation = np.zeros(3, 3)

    # extrinsicMatrix = np.hstack((rotation, translation))

    # Use the extrinsic and the intrinsic matrices to get uv coordinates
    dataFor3Dto2D = rightCameraFrame  # change this to an actual output
    return dataFor3Dto2D

    # FOR DEBUGGING PURPOSES ONLY: Check if dataForFD is a frame from the video captured by client 1
    # frame = np.frombuffer(dataForFD, dtype=np.uint8)
    # print(frame.shape)
    # frame = frame.reshape(w, h, c)
    # cv2.imwrite('Received.png', frame)
    # dataFor3Dto2D = rightCameraFrame # change this to the actual output