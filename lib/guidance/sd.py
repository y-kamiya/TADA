import cv2
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
# from .perpneg_utils import weighted_perpendicular_aggregator


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)  # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, opt):
        super().__init__()

        self.device = device
        self.opt = opt
        self.sd_version = opt.sd_version
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.weighting_strategy = opt.weighting_strategy

        print(f'[INFO] loading stable diffusion...')

        self.resolution = 512
        if opt.hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {opt.hf_key}')
            model_key = opt.hf_key
        elif self.sd_version == '2.1-768':
            model_key = "stabilityai/stable-diffusion-2-1"
            self.resolution = 768
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        pipe_kargs = {
            "use_safetensors": True,
            "load_safety_checker": False,
            "torch_dtype": self.precision_t,
        }

        if opt.ckpt is not None:
            self.pipeline = StableDiffusionPipeline.from_single_file(opt.ckpt).to(self.device)
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_key,
                **pipe_kargs,
            ).to(self.device)

        if opt.lora is not None:
            self.pipeline.load_lora_weights(opt.lora)

        if opt.embeddings is not None:
            self.pipeline.load_textual_inversion(opt.embeddings, token="<V>")

        # Create model
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet

        # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        # self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.scheduler = self.pipeline.scheduler
        self.scheduler.set_timesteps(self.opt.denoise_steps)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
        self.inverse_scheduler.set_timesteps(self.opt.denoise_steps)

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """
        Args:
            prompt: str

        Returns:
            text_embeddings: torch.Tensor
        """
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

    @torch.no_grad()
    def pred_noise(self, latents_noisy, t, text_embeddings, guidance_scale=None):
        latent_model_input = latents_noisy.repeat((2, 1, 1, 1))
        t = t.repeat(2) if t.dim() > 0 else t
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        if guidance_scale is None:
            guidance_scale = self.opt.guidance_scale

        return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    @torch.no_grad()
    def sample_refined_images(self, text_embeddings, pred_rgb, t_annel, sr_model=None, **kwargs):
        pred_rgb_scaled = F.interpolate(pred_rgb, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_scaled)

        t2_schedule_current = self.opt.t2_schedule[0] - t_annel * (self.opt.t2_schedule[0] - self.opt.t2_schedule[1])
        if t2_schedule_current >= 1.0:
            t2_schedule_current = 0.999999
        t1_index = int(t2_schedule_current * self.opt.denoise_steps * self.opt.t1_ratio)
        t1 = self.inverse_scheduler.timesteps[t1_index]
        t2 = t2_schedule_current * self.num_train_timesteps

        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t1)

        for i, t in enumerate(self.inverse_scheduler.timesteps[:-1]):
            t_prev = self.inverse_scheduler.timesteps[i+1]
            if t_prev <= t1:
                continue
            if t2 < t_prev:
                break
            noise_pred = self.pred_noise(latents_noisy, t, text_embeddings, guidance_scale=0)
            latents_noisy = self.inverse_scheduler.step(noise_pred, t_prev, latents_noisy).prev_sample

        for t in self.scheduler.timesteps:
            if t2 < t:
                continue
            noise_pred = self.pred_noise(latents_noisy, t, text_embeddings)
            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy, eta=self.opt.ddim_eta).prev_sample.to(latents.dtype)

        x0 = self.decode_latents(latents_noisy)

        if sr_model is not None:
            x0 = sr_model(x0)

        return F.interpolate(x0, (pred_rgb.shape[-2], pred_rgb.shape[-1]), mode='bicubic', align_corners=False)

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=None, rgb_as_latents=False, data=None, bg_color=None, is_full_body=True):
        if rgb_as_latents:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
            latents = latents * 2 - 1
        else:
            pred_rgb_scaled = F.interpolate(pred_rgb, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_scaled)

        # latents = torch.mean(latents, keepdim=True, dim=0)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self.pred_noise(latents_noisy, t, text_embeddings, guidance_scale)

        # w(t), sigma_t^2
        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, (latents - grad).detach(), reduction="sum") / latents.shape[0]

        return loss

    def produce_latents(self, text_embeddings, height=512, width=512, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                noise_pred = self.pred_noise(latents, t, text_embeddings, guidance_scale)
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        uncon_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]
        text_embeds = torch.cat([uncon_embeds, text_embeds])

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--negative', default='bad anatomy', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--denoise_steps', type=int, default=50)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora', type=str, default=None)
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--weighting_strategy', type=str, default='fantasia3d')
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98])
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, False, opt)

    # subjects = open("./data/prompt/fictional.txt", 'r').read().splitlines()[:5]
    # opt.negative
    imgs = [
        np.vstack([
            # sd.prompt_to_img(f"a 3D rendering of the mouth of {prompt}, {v}", opt.negative, opt.H, opt.W)[0]
            np.hstack([
                sd.prompt_to_img(f"a {v} view 3D rendering of {opt.prompt}, full-body", opt.negative, opt.H, opt.W, guidance_scale=7.5)[0],
                sd.prompt_to_img(f"a {v} view 3D rendering of {opt.prompt}, face", opt.negative, opt.H, opt.W, guidance_scale=7.5)[0],
            ])

            for v in ["front" "side", "back", "overhead"]
            # for v in ["front view" "back view", "side view", "overhead view"]
        ])
        # for prompt in subjects
    ]

    cv2.imwrite("sd.png", np.hstack(imgs)[..., ::-1])

    # visualize image
    # plt.imshow(imgs[0])
    # plt.show()
