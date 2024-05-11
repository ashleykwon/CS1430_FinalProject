# pip install timm==0.6.7
import torch
# from geometry import depth_to_points
from PIL import Image
import numpy as np

def depth_to_points(depth, K, R, t):

    Kinv = np.linalg.inv(K)

    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :]

    return pts3D_2[:, :, :, :3, 0][0]


# image = Image.open("/Users/apoorvkh/Downloads/0.jpg").convert("RGB")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class ZoeProjection:
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(device).eval()

    def to_3d_points(self, pil_image, K, R, t):
        print()
        depth = self.model.infer_pil(pil_image)[None]
        pts3d = depth_to_points(depth, K, R, t)
        pts3d = pts3d.reshape(-1, 3)
        return pts3d

# Tri-mesh
# image = np.array(image)
# verts = pts3d
# triangles = create_triangles(image.shape[0], image.shape[1])
# colors = image.reshape(-1, 3)
# mesh = trimesh.Trimesh(vertices=pts3d, faces=triangles, vertex_colors=colors)
