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

def interpolate_extrinsic_matrices(R_l, t_l, R_r, t_r, new_x, new_y):
    # Derive a new extrinsic matrix for the third camera (user's head) with new_x and new_y
    R_l = R.from_matrix(R_l).as_euler('xyz', degrees=True) # Change this to a Rotation instance
    R_r = R.from_matrix(R_r).as_euler('xyz', degrees=True) # Change this to a Rotation instance
    rots = np.asarray([R_l, R_r])
    rots = R.from_euler('xyz', rots, degrees=True)
    slerp = Slerp([0,1], rots)
    newRotationMatrix = slerp([new_x])[0].as_matrix()
    newTranslationVec = np.multiply(np.asarray([new_x, new_y, 1]), t_r)
    newTranslationVec = np.asarray([[newTranslationVec[0]], [newTranslationVec[1]], [newTranslationVec[2]]])
    return np.hstack((newRotationMatrix, newTranslationVec))

def reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r, new_x, new_y) -> Image.Image:
    # Assume K_l and K_r are the same
    # Assume t_r is a zero vector and R_r is an identity matrix
    # Assume leftCameraFrame and rightCameraFrame have the same shape

    # TODO: can we do this in one pass?
    leftCameraDepth = zoe_depth.get_depth(leftCameraFrame)
    rightCameraDepth = zoe_depth.get_depth(rightCameraFrame)

    leftCameraFrame = np.asarray(leftCameraFrame)
    rightCameraFrame = np.asarray(rightCameraFrame)

    H, W = leftCameraFrame.shape[:2]

    leftCameraTo3D = depth_to_points(
        leftCameraDepth,
        K_l, R_l, t_l
    )
    rightCameraTo3D = depth_to_points(
        rightCameraDepth, 
        K_r, R_r, t_r
    )

    leftCameraTo3D = np.concatenate((
            leftCameraTo3D,
            np.ones(leftCameraTo3D.shape[:2] + (1,))
        ),
        axis=2
    ).reshape(H * W, 4).T

    rightCameraTo3D = np.concatenate((
            rightCameraTo3D,
            np.ones(rightCameraTo3D.shape[:2] + (1,))
        ),
        axis=2
    ).reshape(H * W, 4).T

    intrinsic = K_l
    extrinsic = interpolate_extrinsic_matrices(R_l, t_l, R_r, t_r, new_x, new_y)

    remapped2DCoordsLeft = intrinsic @ extrinsic @ leftCameraTo3D
    remapped2DCoordsLeft[0, :] /= remapped2DCoordsLeft[2, :]
    remapped2DCoordsLeft[1, :] /= remapped2DCoordsLeft[2, :]
    remapped2DCoordsLeft = remapped2DCoordsLeft[:2, :].astype(int)

    remapped2DCoordsRight = intrinsic@extrinsic@rightCameraTo3D # 3 x N
    remapped2DCoordsRight[0, :] /= remapped2DCoordsRight[2, :] 
    remapped2DCoordsRight[1, :] /= remapped2DCoordsRight[2, :] 
    remapped2DCoordsRight = remapped2DCoordsRight[:2, :].astype(int)  # 2 x N

    ###

    # remapped2DCoordsLeft = remapped2DCoordsLeft[::-1, :]
    # remapped2DCoordsRight = remapped2DCoordsRight[::-1, :]

    # valid_mappings = (
    #     ((remapped2DCoordsLeft[0, :] >= 0) & (remapped2DCoordsLeft[0, :] < H)) &
    #     ((remapped2DCoordsLeft[1, :] >= 0) & (remapped2DCoordsLeft[1, :] < W))
    # )

    # leftCameraFrame = leftCameraFrame.reshape(H*W, 3)
    # remapped2DCoordsLeft = np.ravel_multi_index(remapped2DCoordsLeft, (H, W), mode='clip')

    # new_image = np.zeros((H * W, 3), dtype=np.uint8)
    # new_image[valid_mappings] = leftCameraFrame[remapped2DCoordsLeft[valid_mappings]]

    # valid_mappings = (
    #     ((remapped2DCoordsRight[0, :] >= 0) & (remapped2DCoordsRight[0, :] < H)) &
    #     ((remapped2DCoordsRight[1, :] >= 0) & (remapped2DCoordsRight[1, :] < W))
    # )

    # rightCameraFrame = rightCameraFrame.reshape(H*W, 3)
    # remapped2DCoordsRight = np.ravel_multi_index(remapped2DCoordsRight, (H, W), mode='clip')

    # new_image[valid_mappings] = rightCameraFrame[remapped2DCoordsRight[valid_mappings]]
    # new_image = new_image.reshape(H, W, 3)

    ## working

    new_image = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(remapped2DCoordsLeft.shape[1]):
        leftUV = remapped2DCoordsLeft[:,i]
        rightUV = remapped2DCoordsRight[:,i]

        u_l = int(leftUV[1])
        v_l = int(leftUV[0])

        u_r = int(rightUV[1])
        v_r = int(rightUV[0])

        if 0 <= u_l < H and 0 <= v_l < W:
            new_image[u_l, v_l] = leftCameraFrame[u_l, v_l,:]
        if 0 <= u_r < H and 0 <= v_r < W:
            new_image[u_r, v_r] = rightCameraFrame[u_r, v_r,:]

    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    return new_image


if __name__ == "__main__":
    K_l, dist_l, R_l, t_l = pickle.load(open("test_data/left_camera.pickle", 'rb'))
    K_r, dist_r, R_r, t_r = pickle.load(open("test_data/right_camera.pickle", 'rb'))

    t_r = t_r[:, 0]

    leftCameraFrame = Image.open("test_data/left.jpg")
    rightCameraFrame = Image.open("test_data/right.jpg")

    new_x = 0.5
    new_image = reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r, new_x, 0.5)
    cv2.imwrite(f'output.jpg', new_image)

    # for i in range(11):
    #     new_x = i / 10
    #     new_image = reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r, new_x, 0.5)
    #     cv2.imwrite(f'outputs/{i}.jpg', new_image)
