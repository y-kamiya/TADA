import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.provider import ViewDataset
from lib.trainer import *
from lib.dlmesh import DLMesh
from lib.common.utils import load_config

import threestudio.utils.config as three_cfg

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    # parser.add_argument('--mesh', type=str, required=True, help="mesh template, must be obj format")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', help="negative text prompt")
    args = parser.parse_args()

    cfg = load_config(args.config, 'configs/default.yaml')

    cfg.merge_from_list([
        'text', args.text,
        'negative', args.negative,
    ])
    # cfg.model.merge_from_list(['mesh', args.mesh])
    # cfg.training.merge_from_list(['workspace', args.workspace])
    cfg.freeze()

    seed_everything(cfg.seed)

    if cfg.guidance.name == 'imagedream':
        exp_cfg: three_cfg.ExperimentConfig
        exp_cfg = three_cfg.load_config("ImageDream/configs/imagedream-sd21-shading.yaml")


    def build_dataloader(phase):
        """
        Args:
            phase: str one of ['train', 'test' 'val']
        Returns:
        """
        opt = cfg.guidance
        if opt.name == 'imagedream' and phase == "train":
            import threestudio.data.random_multiview as mv
            config = mv.parse_structured(mv.RandomMultiviewCameraDataModuleConfig, exp_cfg.data)
            dataset = mv.RandomMultiviewCameraIterableDataset(config)
            return DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=dataset.collate)
        else:
            batch_size = 1
            if phase == "train":
                batch_size = 2
            size = 4 if phase == 'val' else 100
            dataset = ViewDataset(cfg.data, device=device, type=phase, size=size)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def configure_guidance():
        opt = cfg.guidance
        if opt.name == 'imagedream':
            from lib.guidance.multiview_diffusion import MultiviewDiffusion
            from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
            prompt_processor = StableDiffusionPromptProcessor(exp_cfg.system.prompt_processor)
            return MultiviewDiffusion(exp_cfg.system.guidance, prompt_processor())
        elif opt.name == 'sd':
            from lib.guidance.sd import StableDiffusion
            return StableDiffusion(device, cfg.fp16, opt)
        elif opt.name == 'if':
            from lib.guidance.deepfloyd import IF
            return IF(device, opt.vram_O)
        else:
            from lib.guidance.clip import CLIP
            return CLIP(device)

    def configure_optimizer():
        opt = cfg.training
        if opt.optim == 'adan':
            from lib.common.optimizer import Adan

            optimizer = lambda model: Adan(
                model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else:  # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(5 * opt.lr), betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.1 ** min(x / opt.iters, 1))
        return scheduler, optimizer

    model = DLMesh(cfg.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.test:
        trainer = Trainer(cfg.name,
                          text=cfg.text,
                          negative=cfg.negative,
                          dir_text=cfg.data.dir_text,
                          opt=cfg.training,
                          model=model,
                          guidance=None,
                          device=device,
                          fp16=cfg.fp16
                          )

        test_loader = build_dataloader('test')

        trainer.test(test_loader)

        if cfg.save_mesh:
            trainer.save_mesh()

    else:
        train_loader = build_dataloader('train')

        scheduler, optimizer = configure_optimizer()
        try:
            guidance = configure_guidance()
        except:
            guidance = configure_guidance()
        trainer = Trainer(cfg.name,
                          text=cfg.text,
                          negative=cfg.negative,
                          dir_text=cfg.data.dir_text,
                          opt=cfg.training,
                          model=model,
                          guidance=guidance,
                          device=device,
                          optimizer=optimizer,
                          fp16=cfg.fp16,
                          lr_scheduler=scheduler,
                          scheduler_update_every_step=True
                          )
        trainer.save_config(cfg)

        if os.path.exists(cfg.data.image):
            trainer.default_view_data = train_loader.dataset.get_default_view_data()

        valid_loader = build_dataloader('val')
        max_epoch = np.ceil(cfg.training.iters / (len(train_loader) * train_loader.batch_size)).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # test
        test_loader = build_dataloader('test')
        trainer.test(test_loader)
        trainer.save_mesh()
