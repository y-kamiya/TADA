import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDIMInverseScheduler

from threestudio.models.guidance.multiview_diffusion_guidance import MultiviewDiffusionGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from imagedream.ldm.util import add_random_background


scheduler_config = {
    "_class_name": "DDIMScheduler",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "dynamic_thresholding_ratio": 0.995,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "steps_offset": 1,
    "thresholding": False,
    "timestep_spacing": "leading",
    "trained_betas": None
}


class MultiviewDiffusion(MultiviewDiffusionGuidance):
    def __init__(self, three_cfg, opt, prompt_utils):
        super().__init__(three_cfg)
        self.opt = opt
        self.prompt_utils = prompt_utils
        self.resolution = 256
        self.scheduler = DDIMScheduler.from_config(scheduler_config)
        self.scheduler.set_timesteps(opt.denoise_steps)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
        self.inverse_scheduler.set_timesteps(opt.denoise_steps)

    def get_text_embeds(self, prompts):
        pass

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, rgb_as_latents=False, data=None, bg_color=None, is_full_body=True):
        rgb = pred_rgb.permute(0, 2, 3, 1)
        output = self.forward(
            rgb=rgb,
            prompt_utils=self.prompt_utils,
            comp_rgb_bg=bg_color,
            is_full_body=is_full_body,
            **data,
            # text_embeddings=text_embeddings,
        )
        return output["loss_sds"]

    def build_context(self, text_embeddings, **kwargs):
        camera = self.get_camera_cond(kwargs["c2w"], kwargs["fovy"])
        camera = camera.repeat(2, 1).to(text_embeddings)

        ip_img = self.prompt_image(self.prompt_utils, kwargs["is_full_body"])
        bg_color = kwargs["comp_rgb_bg"].mean().detach().cpu().numpy() * 255
        ip_img = add_random_background(ip_img, bg_color)
        image_embeddings = self.model.get_learned_image_conditioning(ip_img)
        un_image_embeddings = torch.zeros_like(image_embeddings).to(image_embeddings)

        bs = text_embeddings.shape[0] // 2
        ip = torch.cat([
            image_embeddings.repeat(bs, 1, 1),
            un_image_embeddings.repeat(bs, 1, 1)
        ], dim=0).to(text_embeddings)

        return {
            "context": text_embeddings,
            "camera": camera,
            "num_frames": 5,
            "ip": ip,
        }, ip_img

    @torch.no_grad()
    def pred_noise(self, latents_noisy, t, context, ip, guidance_scale=None):
        latent_model_input = latents_noisy.repeat((2, 1, 1, 1))
        t_expand = torch.tensor(t, device=latents_noisy.device).repeat(latent_model_input.shape[0])

        latent_model_input, t_expand, context = self.append_extra_view(latent_model_input, t_expand, context.copy(), ip=ip)
        noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)

        if guidance_scale is None:
            guidance_scale = self.opt.guidance_scale

        return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    @torch.no_grad()
    def sample_refined_images(self, text_embeddings, pred_rgb, t_annel, **kwargs):
        assert kwargs["c2w"] is not None and kwargs["fovy"] is not None

        pred_rgb_scaled = F.interpolate(pred_rgb, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_scaled)

        # t2_schedule_current = self.opt.t2_schedule[0] - t_annel * (self.opt.t2_schedule[0] - self.opt.t2_schedule[1])
        t2_schedule_current = kwargs["t"]
        if t2_schedule_current >= 1.0:
            t2_schedule_current = 0.999999
        t1_index = int(t2_schedule_current * self.opt.denoise_steps * self.opt.t1_ratio)
        # t1_index = torch.tensor(t2_schedule_current * self.opt.denoise_steps * 0.6, dtype=torch.long, device=self.device)
        t1 = self.inverse_scheduler.timesteps[t1_index]
        t2 = t2_schedule_current * self.num_train_timesteps

        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t1)

        text_embeddings = self.prompt_utils.get_text_embeddings(
            kwargs["elevation"], kwargs["azimuth"], kwargs["camera_distances"], False
        )
        context, ip_img = self.build_context(text_embeddings, **kwargs)

        for i, t in enumerate(self.inverse_scheduler.timesteps[:-1]):
            t_prev = self.inverse_scheduler.timesteps[i+1]
            if t_prev <= t1:
                continue
            if t2 < t_prev:
                break
            noise_pred = self.pred_noise(latents_noisy, t, context, ip_img, guidance_scale=0)
            latents_noisy = self.inverse_scheduler.step(noise_pred[:-1], t_prev, latents_noisy).prev_sample

        for t in self.scheduler.timesteps:
            if t2 < t:
                continue
            noise_pred = self.pred_noise(latents_noisy, t, context, ip_img)
            latents_noisy = self.scheduler.step(noise_pred[:-1], t, latents_noisy, eta=0.0).prev_sample.to(latents.dtype)

        x0 = self.decode_latents(latents_noisy)

        return F.interpolate(x0, (pred_rgb.shape[-2], pred_rgb.shape[-1]), mode='bilinear', align_corners=False)

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs
