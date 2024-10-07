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


def main(args):
    start = time.time()

    device = torch.device("cuda")
    radius_list = torch.arange(args.radius_min, args.radius_max, args.radius_step)
    center_diff_list = torch.arange(args.center_min, args.center_max, args.center_step) if args.face else torch.tensor([0])
    H = args.height
    W = args.width

    cfg = load_config('configs/default.yaml')
    model = DLMesh(cfg.model).eval().to(device)
    face_center, face_scale = model.get_mesh_center_scale("face")

    dataset = ZoomOutViewDataset(cfg.data, device=device, radius_list=radius_list, is_full_body=not args.face)
    dataset.face_scale = face_scale.item()
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((H, W)),
        torchvision.transforms.ToTensor()
    ])

    image = Image.open(args.image_path)
    image = transforms(image)
    alpha_ref = image[3:, :, :].unsqueeze(0).to(device)

    min_loss = 1e9
    best_radius = -1.0
    best_center = None
    best_alpha = None
    for center_diff in center_diff_list:
        if args.face:
            center = face_center.clone().detach()
            center[1] += center_diff
            dataset.face_center = center
        for i, data in enumerate(dataloder):
            with torch.no_grad():
                out = model(data["rays_o"], data["rays_d"], data["mvp"], H, W, shading="albedo")
            alpha = out['alpha'].permute(0, 3, 1, 2)

            loss = F.mse_loss(alpha, alpha_ref).item()
            radius = radius_list[i].item()
            diff = center_diff.item()
            if loss < min_loss:
                min_loss = loss
                best_radius = radius
                best_center = diff
                best_alpha = alpha

            print(f"{radius:.3f}, center: {diff:.4f}, loss: {loss:.4f}")
            # output = torch.cat([alpha_ref, alpha])
            # torchvision.utils.save_image(output, f"tmp/{diff:.3f}_{radius:.2f}.jpg")

        print("---")

    print(f"radius: {best_radius:.3f}, center: {best_center:.4f}, loss: {min_loss:.4f}")
    output = torch.cat([alpha_ref, best_alpha])
    torchvision.utils.save_image(output, "best_alpha.jpg")

    print(time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--face", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--radius_min", type=float, default=0.5)
    parser.add_argument("--radius_max", type=float, default=1.6)
    parser.add_argument("--radius_step", type=float, default=0.1)
    parser.add_argument("--center_min", type=float, default=-0.04)
    parser.add_argument("--center_max", type=float, default=0.05)
    parser.add_argument("--center_step", type=float, default=0.01)
    args = parser.parse_args()
    print(args)

    main(args)
