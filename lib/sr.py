import os
import torch
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.rrdbnet_arch import RRDBNet


class RealESRGAN:
    MODELS = {
        2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }

    def __init__(self, device, scale):
        if scale not in self.MODELS:
            raise ValueError(f"model for scale x{scale} does not exist")

        self.rrdbnet = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

        url = self.MODELS[scale]
        name = os.path.basename(url)
        model_path = load_file_from_url(url=url, model_dir="data/realesrgan", progress=True, file_name=name)
        data = torch.load(model_path, map_location="cpu", weights_only=True)
        key = "params_ema" if 'params_ema' in data else "params"
        self.rrdbnet.load_state_dict(data[key], strict=True)

        self.rrdbnet.eval()
        self.rrdbnet = self.rrdbnet.to(device)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor):
        return self.rrdbnet(image)


if __name__ == "__main__":
    from torchvision.utils import save_image
    from torchvision.io import read_image
    from torchvision.transforms.v2.functional import to_dtype

    scale = 2
    dtype = torch.float16
    device_name = "cuda"
    device = torch.device(device_name)

    img = read_image("sr_test/input/unitychan_unique3d.png")
    img = to_dtype(img, scale=True).unsqueeze(0).to(device)
    model = RealESRGAN(device, scale)

    with torch.autocast(device_name, dtype=dtype):
        print(img.shape, img.dtype, img.min(), img.max())
        output = model(img)
        print(output.shape, output.dtype, output.min(), output.max())

    save_image(output, "realesrgan_output.png")
