import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDIMInverseScheduler

from threestudio.models.guidance.multiview_diffusion_guidance import MultiviewDiffusionGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor


class MultiviewDiffusion(MultiviewDiffusionGuidance):
    def __init__(self, three_cfg, opt, prompt_utils):
        super().__init__(three_cfg)
        self.opt = opt
        self.prompt_utils = prompt_utils
        self.resolution = 256
        m = self.model
        self.scheduler = DDIMScheduler(beta_start=m.linear_start, beta_end=m.linear_end)
        self.scheduler.set_timesteps(opt.denoise_steps)
        self.inverse_scheduler = DDIMInverseScheduler(beta_start=m.linear_start, beta_end=m.linear_end)
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

    def build_context(self, text_embeddings, camera, fovy):
        camera = self.get_camera_cond(camera, fovy)
        camera = camera.repeat(2, 1).to(text_embeddings)
        num_frames = 5
        return {
            "context": text_embeddings,
            "camera": camera,
            "num_frames": num_frames, # number of frames
        }

    @torch.no_grad()
    def pred_noise(self, latents_noisy, t, context, guidance_scale=None):
        latent_model_input = latents_noisy.repeat((2, 1, 1, 1))
        t = t.repeat(2) if t.dim() > 0 else t

        latent_model_input, t, context = self.append_extra_view(latent_model_input, t, context, ip=ip)
        noise_pred = self.model.apply_model(latent_model_input, t, context)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        if guidance_scale is None:
            guidance_scale = self.opt.guidance_scale

        return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    @torch.no_grad()
    def sample_refined_images(self, text_embeddings, pred_rgb, t_annel, **kargs):
        assert kargs["c2w"] is not None and kargs["fovy"] is not None

        pred_rgb_scaled = F.interpolate(pred_rgb, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_scaled)

        # t2_schedule_current = self.opt.t2_schedule[0] - t_annel * (self.opt.t2_schedule[0] - self.opt.t2_schedule[1])
        t2_schedule_current = kargs["t"]
        t1_index = torch.tensor(t2_schedule_current * self.opt.denoise_steps * 0.6, dtype=torch.long, device=self.device)
        t1 = self.inverse_scheduler.timesteps[t1_index]
        t2 = t2_schedule_current * self.num_train_timesteps

        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t1)

        context = self.build_context(text_embeddings, kargs["c2w"], kargs["fovy"])

        for i, t in enumerate(self.inverse_scheduler.timesteps[:-1]):
            t_prev = self.inverse_scheduler.timesteps[i+1]
            if t_prev <= t1:
                continue
            if t2 < t_prev:
                break
            noise_pred = self.pred_noise(latents_noisy, t, context, guidance_scale=0)
            latents_noisy = self.inverse_scheduler.step(noise_pred, t_prev, latents_noisy).prev_sample

        for t in self.scheduler.timesteps:
            if t2 < t:
                continue
            noise_pred = self.pred_noise(latents_noisy, t, context)
            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy, eta=1.0).prev_sample.to(latents.dtype)

        x0 = self.model.decode_first_stage(latents_noisy)

        return F.interpolate(x0, (pred_rgb.shape[-2], pred_rgb.shape[-1]), mode='bilinear', align_corners=False)

