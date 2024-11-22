import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch import linalg as LA

from threestudio.models.mesh import Mesh


class Sobel(nn.Module):
    def __init__(self, factor=1):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 8. * factor
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 8. * factor
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)])
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        return LA.norm(x, dim=1, keepdim=True)


class BilateralSmoother:
    def __init__(self, device):
        self.sobel = Sobel().to(device)
        self.transform = v2.Grayscale(1)

    def __call__(self, image, normal, alpha):
        image_gray = self.transform(image)
        image_sobel = self.sobel(image_gray)

        b, c, h, w = normal.shape
        normal_sobel = self.sobel(normal.reshape(b * c, 1, h, w))
        normal_sobel = LA.norm(normal_sobel.view(b, c, h, w), dim=1, keepdim=True)

        loss_normal = torch.sqrt(1 + normal_sobel) * torch.exp(-3. * image_sobel.detach()) * alpha.detach()
        loss_rgb = torch.sqrt(1 + image_sobel) * torch.exp(-3. * image_sobel.detach()) * alpha.detach()
        return loss_normal + loss_rgb


class MeshRegularizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, pr_mesh):
        v_pos = pr_mesh.v
        t_pos_idx = pr_mesh.f.to(dtype=torch.int64)
        mesh = Mesh(v_pos, t_pos_idx)

        # avoid sparse to support fp16
        loss_lap = 0.
        if 0. < self.cfg.lambda_laplacian:
            L = mesh._laplacian_uniform().to_dense()
            loss_lap = L.mm(v_pos).norm(dim=1).mean()

        loss_nc = 0.
        if 0. < self.cfg.lambda_normal_consistency:
            loss_nc = mesh.normal_consistency()

        loss_expand = 0.
        if 0. < self.cfg.lambda_expand:
            loss_expand = 0.5 * F.mse_loss(v_pos, (v_pos + pr_mesh.vn).detach()).mean()

        return self.cfg.lambda_laplacian * loss_lap \
                + self.cfg.lambda_normal_consistency * loss_nc \
                + self.cfg.lambda_expand * loss_expand


class PixelRegularizer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.bilateral = BilateralSmoother(device)

    def __call__(self, image, normal, alpha):
        if 0. < self.cfg.lambda_bilateral:
            return self.cfg.lambda_bilateral * self.bilateral(image, normal, alpha).mean()

        return 0.
