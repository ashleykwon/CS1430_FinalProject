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
    h, w, c = leftCamFrameNP.shape
    leftCamFrameFlat = leftCamFrameNP.reshape(c, h*w) # RGB values from the left camera frame
    rightCamFrameFlat = rightCamFrameNP.reshape(c, h*w) # RGB values from the right camera frame

    # Derive a new extrinsic matrix for the third camera (user's head) with new_x and new_y
    R_l = R.from_matrix(R_l).as_euler('xyz', degrees=True) # Change this to a Rotation instance
    R_r = R.from_matrix(R_r).as_euler('xyz', degrees=True) # Change this to a Rotation instance
    rots = np.asarray([R_l, R_r])
    rots = R.from_euler('xyz', rots, degrees=True)
    slerp = Slerp([0,1], rots)
    newRotationMatrix = slerp([new_x])[0].as_matrix()
    newTranslationVec = np.multiply(np.asarray([new_x, new_y, 1]), t_r)
    newTranslationVec = np.asarray([[newTranslationVec[0]], [newTranslationVec[1]], [newTranslationVec[2]]])

    H, W = leftCameraTo3D.shape[:2]

    leftCameraTo3D = np.concatenate((
            leftCameraTo3D,
            np.ones(leftCameraTo3D.shape[:2] + (1,))
        ),
        axis=2
    ).T.reshape(4, -1)

    rightCameraTo3D = np.concatenate((
            rightCameraTo3D,
            np.ones(rightCameraTo3D.shape[:2] + (1,))
        ),
        axis=2
    ).T.reshape(4, -1)

    intrinsic = K_r
    extrinsic = np.hstack((newRotationMatrix, newTranslationVec))
    remapped2DCoordsLeft = intrinsic @ extrinsic @ leftCameraTo3D
    remapped2DCoordsLeft[0, :] /= remapped2DCoordsLeft[2, :]
    remapped2DCoordsLeft[1, :] /= remapped2DCoordsLeft[2, :]

    remapped2DCoordsRight = intrinsic@extrinsic@rightCameraTo3D # 3 by N
    remapped2DCoordsRight[0, :] /= remapped2DCoordsRight[2, :] 
    remapped2DCoordsRight[1, :] /= remapped2DCoordsRight[2, :] 

    new_image = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(remapped2DCoordsLeft.shape[1]):
        leftUV = remapped2DCoordsLeft[:,i]
        rightUV = remapped2DCoordsRight[:,i]
        u_l = int(leftUV[1])
        v_l = int(leftUV[0])

        u_r = int(rightUV[1])
        v_r = int(rightUV[0])

        if 0 <= u_l < H and 0 <= v_l < W:
            new_image[u_l, v_l] = leftCamFrameNP[u_l, v_l,:]
        if 0 <= u_r < H and 0 <= v_r < W:
            new_image[u_r, v_r] = rightCamFrameNP[u_r, v_r,:]

    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    return new_image


if __name__ == "__main__":
    K_l, dist_l, R_l, t_l = pickle.load(open("test_data/left_camera.pickle", 'rb'))
    K_r, dist_r, R_r, t_r = pickle.load(open("test_data/right_camera.pickle", 'rb'))

    t_r = t_r[:, 0]

    leftCameraFrame = Image.open("test_data/left.jpg")
    rightCameraFrame = Image.open("test_data/right.jpg")

    for i in range(11):
        new_x = i / 10
        new_image = reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r, new_x, 0.5)
        cv2.imwrite(f'outputs/{i}.jpg', new_image)
