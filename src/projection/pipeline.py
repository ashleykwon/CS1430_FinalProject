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

    newTranslationVec = t_r - t_l
    newTranslationVec[0] *= new_x
    newTranslationVec[1] *= new_y
    newTranslationVec += t_l
    newTranslationVec = newTranslationVec[:, None]

    return np.hstack((newRotationMatrix, newTranslationVec))

def reproject_to_2d(intrinsic, extrinsic, pts_3d, camera_frame, new_image):
    remapped2DCoords = intrinsic @ extrinsic @ pts_3d # 3 x N
    remapped2DCoords[0, :] /= remapped2DCoords[2, :] 
    remapped2DCoords[1, :] /= remapped2DCoords[2, :] 
    remapped2DCoords = remapped2DCoords[:2, :].astype(int)  # 2 x N
    remapped2DCoords = remapped2DCoords[::-1, :]

    H, W, _ = camera_frame.shape

    valid_mappings = (
        ((remapped2DCoords[0, :] >= 0) & (remapped2DCoords[0, :] < H)) &
        ((remapped2DCoords[1, :] >= 0) & (remapped2DCoords[1, :] < W))
    )

    camera_frame_for_mapping = camera_frame.reshape(H*W, 3)
    remapped2DCoords = np.ravel_multi_index(remapped2DCoords, (H, W), mode='clip')
    new_image[valid_mappings] = camera_frame_for_mapping[remapped2DCoords[valid_mappings]]

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

    ## interpolating camera matrices

    intrinsic = K_l
    extrinsic = np.hstack((R_l, t_l[:, None]))
    extrinsic[0, 3] += new_x

    # intrinsic = K_l
    # extrinsic = interpolate_extrinsic_matrices(R_l, t_l, R_r, t_r, new_x, new_y)

    ##

    new_image = np.zeros((H * W, 3), dtype=np.uint8)

    # print(t_l[0], t_r[0], extrinsic[0, 3])

    if abs(t_l[0] - extrinsic[0, 3]) > abs(t_r[0] - extrinsic[0, 3]):
        reproject_to_2d(intrinsic, extrinsic, rightCameraTo3D, rightCameraFrame, new_image)
        reproject_to_2d(intrinsic, extrinsic, leftCameraTo3D, leftCameraFrame, new_image)
    else:
        reproject_to_2d(intrinsic, extrinsic, leftCameraTo3D, leftCameraFrame, new_image)
        reproject_to_2d(intrinsic, extrinsic, rightCameraTo3D, rightCameraFrame, new_image)

    new_image = new_image.reshape(H, W, 3)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    new_image = np.flip(new_image, axis=0)
    new_image = np.flip(new_image, axis=1)

    return new_image


if __name__ == "__main__":
    K_l, dist_l, R_l, t_l = pickle.load(open("test_data/left_camera.pickle", 'rb'))
    K_r, dist_r, R_r, t_r = pickle.load(open("test_data/right_camera.pickle", 'rb'))

    t_l = -1 * t_l
    t_r = -1 * t_r[:, 0]

    leftCameraFrame = Image.open("test_data/left.jpg")
    rightCameraFrame = Image.open("test_data/right.jpg")

    for i in range(11):
        new_x = i * 0.04
        new_image = reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r, new_x, 0.5)
        cv2.imwrite(f'outputs/{i}.jpg', new_image)
