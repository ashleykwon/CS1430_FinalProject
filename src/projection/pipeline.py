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

    # Sanity check to see if we can project leftCameraTo3D and rightCameraTo3D back to 2D images. Comment this out when actually running the code
    leftCamExtrinsicMatrix = np.hstack((R_l, t_l))
    rightCamExtrinsicMatrix = np.hstack((R_r, t_r))
    leftCamFrameReconstructed = np.linalg.inv(K_l)@np.linalg.inv(leftCamExtrinsicMatrix)@leftCameraTo3D
    rightCamFrameReconstructed = np.linalg.inv(K_r)@np.linalg.inv(rightCamExtrinsicMatrix)@rightCameraTo3D
    cv2.imwrite('left_frame_reconstructed.jpg', )


    # TODO 3: Do the 3D to 2D mapping + viewing angle modification based on face detection and save the result in dataFor3Dto2D


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