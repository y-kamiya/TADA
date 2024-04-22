from threestudio.models.guidance.multiview_diffusion_guidance import MultiviewDiffusionGuidance
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor


class MultiviewDiffusion(MultiviewDiffusionGuidance):
    def __init__(self):
        self.prompt_utils = StableDiffusionPromptProcessor({})

    def parameters(self):
        return []

    def get_text_embeds(self, prompts):
        pass

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, rgb_as_latents=False, data=None, bg_color=None):
        output = self.forward(
            rgb=pred_rgb,
            prompt_utils=self.prompt_utils,
            elevation=data["polar"],
            azimuth=data["azimuth"],
            camera_distances=None,
            c2w=data["poses"],
            comp_rgb_bg=bg_color,
            text_embeddings=text_embeddings,
        )
        return output["loss_sds"]
