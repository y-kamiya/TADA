import torch
import torch.nn as nn
import torch.nn.functional as F


class Guidance:
    def __init__(self, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt = opt

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * opt.t_range[1])

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, data=None, bg_color=None, is_full_body=True):
        pred_rgb_scaled = F.interpolate(pred_rgb, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_scaled)

        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
        context = self.build_context(text_embeddings, **data)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self.pred_noise(latents_noisy, t, context, guidance_scale)

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

    @torch.no_grad()
    def encode_images(self, images):
        raise NotImplementedError

    @torch.no_grad()
    def decode_latents(self, latents):
        raise NotImplementedError

    def build_context(self, text_embeddings, **kwargs):
        raise NotImplementedError

    def build_guidance_scale(self):
        return None

    @torch.no_grad()
    def pred_noise(self, latents_noisy, t, context, guidance_scale=None):
        raise NotImplementedError

    @torch.no_grad()
    def sample_refined_images(self, text_embeddings, pred_rgb, t_anneal, sr_model=None, **kwargs):
        pred_rgb_scaled = F.interpolate(pred_rgb, (self.resolution, self.resolution), mode='bilinear', align_corners=False)
        latents = self.encode_images(pred_rgb_scaled)

        t2_schedule_current = self.opt.t2_schedule[0] - t_anneal * (self.opt.t2_schedule[0] - self.opt.t2_schedule[1])
        if t2_schedule_current >= 1.0:
            t2_schedule_current = 0.999999
        t1_index = int(t2_schedule_current * self.opt.denoise_steps * self.opt.t1_ratio)
        idx = self.opt.denoise_steps - t1_index
        t1 = self.scheduler.timesteps[(idx-1):idx]
        t2 = t2_schedule_current * self.num_train_timesteps

        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t1)

        context = self.build_context(text_embeddings, **kwargs)

        if self.inverse_scheduler is not None:
            for i, t in enumerate(self.inverse_scheduler.timesteps[:-1]):
                t_prev = self.inverse_scheduler.timesteps[i+1]
                if t_prev <= t1:
                    continue
                if t2 < t_prev:
                    break
                noise_pred = self.pred_noise(latents_noisy, t, context, guidance_scale=0, is_inverse=True)
                latents_noisy = self.inverse_scheduler.step(noise_pred, t_prev, latents_noisy).prev_sample

        guidance_scale = self.build_guidance_scale()
        for t in self.scheduler.timesteps:
            if t2 < t:
                continue
            noise_pred = self.pred_noise(latents_noisy, t, context, guidance_scale)
            latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy).prev_sample.to(latents.dtype)
            # latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy, eta=self.opt.ddim_eta).prev_sample.to(latents.dtype)

        x0 = self.decode_latents(latents_noisy)

        if sr_model is not None:
            x0 = sr_model(x0)

        return F.interpolate(x0, (pred_rgb.shape[-2], pred_rgb.shape[-1]), mode='bilinear', align_corners=False)

