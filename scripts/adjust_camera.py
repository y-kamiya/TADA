import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from lib.provider import ZoomOutViewDataset, near_head_poses
from lib.dlmesh import DLMesh
from lib.common.utils import load_config


device = torch.device("cpu")
H, W = 512, 512
radius_list = torch.arange(0.5, 1.5, 0.1)

cfg = load_config("adjust_camera.yaml", 'configs/default.yaml')
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
alpha_ref = image[3:, :, :]

losses = []
for i, data in enumerate(dataloder):
    with torch.no_grad():
        out = model(data["rays_o"], data["rays_d"], data["mvp"], H, W, shading="albedo")
    alpha = out['alpha'].permute(0, 3, 1, 2)

    loss = F.mse_loss(alpha, alpha_ref)
    losses.append(loss.item())

tensor = torch.tensor(losses)
idx = tensor.argmin()
optimized_raidus = radius_list[idx]
print(optimized_raidus)
