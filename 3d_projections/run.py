# pip install timm==0.6.7
import torch
from geometry import depth_to_points
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

image = Image.open("/Users/apoorvkh/Downloads/0.jpg").convert("RGB")
depth = model.infer_pil(image)

import cv2
import matplotlib.pyplot as plt
# print(depth)
# exit()
# cv2.imwrite('depth.png', depth)
plt.imshow(depth)
plt.show()

pts3d = depth_to_points(depth[None]).reshape(-1, 3)

print(pts3d.shape)

# Tri-mesh
# image = np.array(image)
# verts = pts3d
# triangles = create_triangles(image.shape[0], image.shape[1])
# colors = image.reshape(-1, 3)
# mesh = trimesh.Trimesh(vertices=pts3d, faces=triangles, vertex_colors=colors)
