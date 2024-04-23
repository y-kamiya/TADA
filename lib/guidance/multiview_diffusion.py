from threestudio.models.guidance.multiview_diffusion_guidance import MultiviewDiffusionGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor


class MultiviewDiffusion(MultiviewDiffusionGuidance):
    def __init__(self, cfg, prompt_utils):
        super().__init__(cfg)
        self.prompt_utils = prompt_utils

    # def parameters(self):
    #     return []

    def get_text_embeds(self, prompts):
        pass

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, rgb_as_latents=False, data=None, bg_color=None):
        rgb = pred_rgb.permute(0, 2, 3, 1)
        output = self.forward(
            rgb=rgb,
            prompt_utils=self.prompt_utils,
            comp_rgb_bg=bg_color,
            **data,
            # text_embeddings=text_embeddings,
        )
        return output["loss_sds"]
