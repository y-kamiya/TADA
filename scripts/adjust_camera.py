import sys
import argparse
import time
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from pathlib import Path

from lib.provider import ZoomOutViewDataset
from lib.dlmesh import DLMesh
from lib.common.utils import load_config
from lib.dpt import DepthNormalEstimation


def calc_y_pad(alpha, threshold=0.8):
    img = alpha[0, 0, :, :]
    H, _ = img.shape
    is_bg = torch.any(img > threshold, dim=1) == False

    mid = int(H // 2)
    upper = is_bg[:mid].sum()
    bottom = is_bg[mid:].sum()

    return upper, bottom


def create_model(args, cfg, device):
    model = DLMesh(cfg.model).eval().to(device)
    if args.ckpt_path is None:
        return model

    ckpt_dict = torch.load(args.ckpt_path, map_location=device)
    if "model" not in ckpt_dict:
        model.load_state_dict(ckpt_dict)
        return model

    model.load_state_dict(ckpt_dict["model"])
    return model


def main(args):
    start = time.time()

    device = torch.device("cuda")
    radius_list = torch.arange(args.radius_min, args.radius_max, args.radius_step)
    center_diff_list = torch.arange(args.center_min, args.center_max, args.center_step) if args.face else torch.tensor([0])
    H = args.height
    W = args.width

    if args.config is None:
        cfg = load_config('configs/default.yaml')
    else:
        cfg = load_config(args.config, 'configs/default.yaml')

    model = create_model(args, cfg, device)
    face_center, face_scale = model.get_mesh_center_scale("face")

    dpt = DepthNormalEstimation(use_depth=False)

    dataset = ZoomOutViewDataset(cfg.data, device=device, radius_list=radius_list, is_full_body=not args.face)
    dataset.face_scale = face_scale.item()
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((H, W)),
        torchvision.transforms.ToTensor()
    ])

    image_ref = Image.open(args.image_path)
    image_ref = transforms(image_ref).unsqueeze(0).to(device)
    alpha_ref = image_ref[:, 3:, :, :]
    image_ref = image_ref[:, :3, :, :] * alpha_ref + (1 - alpha_ref)
    normal_ref = dpt(image_ref)
    normal_ref = (1 - normal_ref) * alpha_ref + (1 - alpha_ref)

    upper_ref, bottom_ref = calc_y_pad(alpha_ref)

    min_loss = 1e9
    best_radius = -1.0
    best_center = None
    best_image = None
    best_normal = None
    best_alpha = None
    center_diff_list = torch.tensor([0.])
    for center_diff in center_diff_list:
        if args.face:
            center = face_center.clone().detach()
            center[1] += center_diff
            dataset.face_center = center
        for i, data in enumerate(dataloder):
            with torch.no_grad():
                out = model(data["rays_o"], data["rays_d"], data["mvp"], H, W, shading="albedo")
            image = out['image'].permute(0, 3, 1, 2)
            normal = out['normal'].permute(0, 3, 1, 2)
            alpha = out['alpha'].permute(0, 3, 1, 2)

            if args.loss_type == "rgb":
                loss = F.mse_loss(image, image_ref).item()
            elif args.loss_type == "normal":
                loss = F.mse_loss(normal, normal_ref).item()
            elif args.loss_type == "alpha":
                loss = F.mse_loss(alpha, alpha_ref).item()
            elif args.loss_type == "pad":
                upper, bottom = calc_y_pad(alpha)
                loss = (torch.abs(upper - upper_ref) + torch.abs(bottom - bottom_ref)).item()
            else:
                raise ValueError(f"loss_type: {args.loss_type} is not defined")

            radius = radius_list[i].item()
            diff = center_diff.item()
            if loss < min_loss:
                min_loss = loss
                best_radius = radius
                best_center = diff
                best_image = image
                best_normal = normal
                best_alpha = alpha

            print(f"{radius:.3f}, center: {diff:.4f}, loss: {loss:.4f}")
            # output = torch.cat([alpha_ref, alpha])
            # torchvision.utils.save_image(output, f"tmp/{diff:.3f}_{radius:.2f}.jpg")

        print("---")

    print(f"radius: {best_radius:.3f}, center: {best_center:.4f}, loss: {min_loss:.4f}")
    output = torch.cat([best_image, image_ref, best_normal, normal_ref, best_alpha.repeat(1,3,1,1), alpha_ref.repeat(1,3,1,1)])
    torchvision.utils.save_image(output, "best_alpha.jpg")

    print(f"elapsed tiem: {time.time() - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--loss_type", default="pad", choices=["rgb", "normal", "alpha", "pad"])
    parser.add_argument("--face", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--radius_min", type=float, default=0.5)
    parser.add_argument("--radius_max", type=float, default=1.8)
    parser.add_argument("--radius_step", type=float, default=0.05)
    parser.add_argument("--center_min", type=float, default=-0.04)
    parser.add_argument("--center_max", type=float, default=0.05)
    parser.add_argument("--center_step", type=float, default=0.01)
    args = parser.parse_args()
    print(args)

    main(args)
