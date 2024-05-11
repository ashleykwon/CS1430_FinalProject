import torch
from PIL import Image

class ZoeDepth:
    def __init__(self, device = 'cpu'):
        self.model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(device).eval()

    def get_depth(self, pil_image: Image.Image):
        return self.model.infer_pil(pil_image)[None]
