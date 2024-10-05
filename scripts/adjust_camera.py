import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from lib.provider import ZoomOutViewDataset
from lib.dlmesh import DLMesh
from lib.common.utils import load_config

import time


start = time.time()

device = torch.device("cuda")
H, W = 1024, 1024
radius_list = torch.arange(0.5, 1.5, 0.1)

cfg = load_config('configs/default.yaml')
model = DLMesh(cfg.model).eval().to(device)

dataset = ZoomOutViewDataset(cfg.data, device=device, radius_list=radius_list)
dataloder = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

transforms = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor()
])
image_path = "data/images/unitychan_a_e0/0.png"
image = Image.open(image_path)
image = transforms(image)
alpha_ref = image[3:, :, :].to(device)

min_loss = 1e9
best_radius = -1.0
best_alpha = None
for i, data in enumerate(dataloder):
    with torch.no_grad():
        out = model(data["rays_o"], data["rays_d"], data["mvp"], H, W, shading="albedo")
    alpha = out['alpha'].permute(0, 3, 1, 2)

    loss = F.mse_loss(alpha, alpha_ref).item()
    if loss < min_loss:
        min_loss = loss
        best_radius = radius_list[i]
        best_alpha = alpha

    print(i, radius_list[i], loss)

print(best_radius)
save_image(best_alpha, "best_alpha.jpg")

print(time.time() - start)
