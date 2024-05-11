# pip install timm==0.6.7
import torch
from geometry import depth_to_points
from PIL import Image

image = Image.open("/Users/apoorvkh/Downloads/0.jpg").convert("RGB")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ZoeProjection:
    def __init__(self, device = 'cpu'):
        self.model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(device).eval()

    def to_3d_points(self, pil_image, K, R, t):
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
