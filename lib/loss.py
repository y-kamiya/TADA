import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from torch import linalg as LA


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


class Smoother():
    def __init__(self, device):
        self.sobel = Sobel().to(device)
        self.transform = v2.Grayscale(1)

    def __call__(self, image, normal, alpha):
        image_gray = self.transform(image)
        image_sobel = self.sobel(image_gray)

        b, c, h, w = normal.shape
        normal_sobel = self.sobel(normal.reshape(b * c, 1, h, w))
        normal_sobel = LA.norm(normal_sobel.view(b, c, h, w), dim=1, keepdim=True)

        return torch.sqrt(1 + normal_sobel) * torch.exp(-3. * image_sobel) * alpha
