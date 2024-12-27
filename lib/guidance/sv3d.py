import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from pathlib import Path
from PIL import Image
import numpy as np

import lib.guidance.utils as utils
from lib.guidance.guidance import Guidance
from lib.guidance.diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline


class StableVideo3d(Guidance):
    def __init__(self, device, fp16, opt):
        super().__init__(opt)

        self.device = device
        self.opt = opt
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.generator = torch.manual_seed(opt.seed)

        print(f"[INFO] loading sv3d ...")

        pipe_kargs = {
            "use_safetensors": True,
            "load_safety_checker": False,
            "torch_dtype": self.precision_t,
        }

        model_key = "chenguolin/sv3d-diffusers"
        self.unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(model_key, subfolder="unet", **pipe_kargs)
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", **pipe_kargs)
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler", **pipe_kargs)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_key, subfolder="image_encoder")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(model_key, subfolder="feature_extractor")

        self.pipeline = StableVideo3DDiffusionPipeline(
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor, 
            unet=self.unet,
            vae=self.vae,
            scheduler=self.scheduler,
        ).to(self.device)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])

        self.inverse_scheduler = EulerAncestralDiscreteScheduler.from_config(self.scheduler.config)
        self.inverse_scheduler.set_timesteps(self.opt.denoise_steps)

        print(f"[INFO] loaded sv3d")

    def build_context(self, _, **kwargs):
        image_embeddings = self.pipeline._encode_image(
            image=kwargs["image"],
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        image = pipeline.video_processor.preprocess(kwargs["image"], height=height, width=width).to(self.device)
        noise = randn_tensor(image.shape, generator=self.generator, device=self.device, dtype=image.dtype)
        noise_aug_strength = 1e-5
        image = image + noise_aug_strength * noise
        image_latents = self.encode_images(image)

        added_time_ids = self.pipeline._get_add_time_ids(
            noise_aug_strength=noise_aug_strength,
            polars_rad=kwargs["polars_rad"],
            azimuths_rad=kwargs["azimuths_rad"],
            dtype=image_embeddings.dtype,
            batch_size=1,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        added_time_ids = [a.to(self.device) for a in added_time_ids]  # (cond_aug, polars_rad, azimuths_rad)

        return {
            "image_embeddings": image_embeddings,
            "image_latents": image_latents,
            "added_time_ids": added_time_ids,
        }

    @torch.no_grad()
    def pred_noise(self, latents_noisy, t, context, guidance_scale=None):
        t = t.repeat(2) if t.dim() > 0 else t
        latent_model_input = self.scheduler.scale_model_input(latents_noisy, t)

        # TODO broadcast?
        latent_model_input = torch.cat([latent_model_input, context["image_latents"]], dim=2)
        latent_model_input = torch.cat([latent_model_input] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=context["image_embeddings"],
            added_time_ids=context["added_time_ids"],
            return_dict=False,
        )[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        if guidance_scale is None:
            guidance_scale = self.opt.guidance_scale

        return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    def encode_images(self, images):
        latents = self._encode_vae_image(
            images,
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
        ).to(self.precision_t)
        return latents.unsqueeze(1).repeat(1, images.shape[0], 1, 1, 1)

    @torch.no_grad()
    def decode_latents(self, latents):
        videos = pipeline.decode_latents(latents, latents.shape[1], decode_chunk_size=1)
        videos = self.video_processor.postprocess_video(video=frames, output_type="pt")
        return videos[0]


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=Path)
    parser.add_argument('--elevation', type=int, default=0)
    parser.add_argument('-H', type=int, default=576)
    parser.add_argument('-W', type=int, default=576)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--denoise_steps', type=int, default=25)
    # to avoid error
    parser.add_argument('--weighting_strategy', type=str, default='fantasia3d')
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98])
    opt = parser.parse_args()

    utils.seed_everything(opt.seed)

    device = torch.device('cuda')

    sv3d = StableVideo3d(device, True, opt)

    num_frames, sv3d_res = 21, 576
    elevations_deg = [opt.elevation] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()

    with (torch.no_grad(),
          torch.autocast("cuda", dtype=torch.float16, enabled=True)):
        image = Image.open(opt.image_path)
        image.load()  # required for `.split()`
        if len(image.split()) == 4:  # RGBA
            input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white bg
            input_image.paste(image, mask=image.split()[3])  # 3rd is the alpha channel
        else:
            input_image = image

        video_frames = sv3d.pipeline(
            input_image.resize((sv3d_res, sv3d_res)),
            height=sv3d_res,
            width=sv3d_res,
            num_frames=num_frames,
            decode_chunk_size=1,  # smaller to save memory
            polars_rad=polars_rad,
            azimuths_rad=azimuths_rad,
            generator=torch.manual_seed(opt.seed),
        ).frames[0]

    name = opt.image_path.name
    fps = 7
    video_frames[0].save(
        "sv3d.gif",
        save_all=True,
        append_images=video_frames[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
    )


