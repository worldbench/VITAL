"""Standalone VITAL inference with separately downloaded public base weights."""
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from vital.vendor.wan.simwam import SimWAM
from vital.vendor.wan.action_dit import ActionDiT
from vital.vendor.wan.wan_video_dit import WanVideoDiT
from vital.vendor.wan.wan_video_vae import WanVideoVAE38
from vital.vendor.wan.mot import MoT
from .f0_adapter import configure_f0_kv_adapter, load_f0_adapter_state_dict, set_f0_prefix_tokens
from .decoder import TrajectoryDecoder
from .residual import ResidualPolicy
from .residual_math import build_features, compose_trajectory
from vital.weights import UPSTREAM, load_bundle, verify_file


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


class VITALModel(SimWAM):
    def _build_mot_attention_mask(self, video_seq_len, action_seq_len,
                                  video_tokens_per_frame, device):
        set_f0_prefix_tokens(self, int(video_tokens_per_frame))
        return super()._build_mot_attention_mask(video_seq_len, action_seq_len,
                                                video_tokens_per_frame, device)


class VITALPolicy:
    def __init__(self, weights='checkpoints/vital.pt', device='cuda',
                 simwam_weights='checkpoints/upstream/SimWAM.pt',
                 wan_vae='checkpoints/upstream/Wan2.2_VAE.pth'):
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the documented inference configuration')
        self.device = device
        bundle = load_bundle(weights)
        config = bundle['model_config']
        # Pin the actual trained visual backbone, not an unadapted WAN DiT.
        verify_file(simwam_weights, UPSTREAM['simwam'])
        verify_file(wan_vae, UPSTREAM['wan_vae'])
        dtype = torch.bfloat16
        # Construction on CPU avoids transient GPU allocation of random FP32 weights.
        video = WanVideoDiT(**config['video']).to(dtype=dtype)
        action = ActionDiT(**config['action']).to(dtype=dtype)
        vae = WanVideoVAE38().to(dtype=dtype)
        base = load(simwam_weights)
        video.load_state_dict({key.removeprefix('mixtures.video.'): value
            for key, value in base['mot'].items()
            if key.startswith('mixtures.video.')}, strict=True)
        del base
        vae_state = load(wan_vae)
        vae_state = vae_state.get('model_state', vae_state)
        vae.load_state_dict({'model.' + key: value for key, value in vae_state.items()}, strict=True)
        del vae_state
        mot = MoT(mixtures={'video': video, 'action': action}, mot_checkpoint_mixed_attn=False)
        self.model = VITALModel(video_expert=video, action_expert=action, mot=mot, vae=vae,
                               text_dim=4096, proprio_dim=8, device='cpu', torch_dtype=dtype)
        core = bundle['action_policy']
        configure_f0_kv_adapter(self.model, config['f0_adapter'])
        action.load_state_dict(core['action_expert'], strict=True)
        load_f0_adapter_state_dict(self.model, core['f0_adapter'])
        self.model.proprio_encoder.load_state_dict(core['proprio_encoder'], strict=True)
        self.mean = core['action_normalizer']['mean'].float().to(device)
        self.std = core['action_normalizer']['std'].float().to(device)
        self.context = core['static_context'].to(dtype=dtype)
        self.context_mask = core['static_context_mask'].bool()
        self.model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.model.device = torch.device(device)
        self.decoder = TrajectoryDecoder().to(device).eval().requires_grad_(False)
        self.decoder.load_state_dict(bundle['trajectory_decoder'], strict=True)
        residual = bundle['residual_policy']
        self.residual = ResidualPolicy(residual['state_dict'], residual['config'])

    @torch.inference_mode()
    def predict(self, rgb, proprio):
        image = TF.to_tensor(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))
        image = TF.resize(image, [384, 672], interpolation=TF.InterpolationMode.BILINEAR,
                          antialias=True)
        image = TF.normalize(image, [0.5] * 3, [0.5] * 3).contiguous()
        proprio = torch.as_tensor(proprio, dtype=torch.float32)
        if proprio.shape != (8,) or not torch.isfinite(proprio).all():
            raise ValueError('Expected velocity[2], acceleration[2], driving_command[4]')
        hidden = []
        hook = self.model.action_expert.head.register_forward_pre_hook(
            lambda _module, inputs: hidden.append(inputs[0].detach()))
        try:
            normalized = self.model.infer_action(prompt=None, input_image=image.unsqueeze(0),
                action_horizon=4, proprio=proprio, context=self.context,
                context_mask=self.context_mask, num_inference_steps=1, seed=0,
                rand_device='cpu', tiled=False)['action']
        finally:
            hook.remove()
        if len(hidden) != 1:
            raise RuntimeError('NFE=1 contract violated')
        tokens = normalized.to(self.device).float()[None] * self.std + self.mean
        base = self.decoder(tokens)[0].cpu().float().numpy()
        features = build_features(normalized.numpy(), base,
            hidden=hidden[0][0].cpu().float().numpy(), ego=proprio.numpy(), kind='hidden')
        delta = self.residual.correction(torch.from_numpy(features)[None])[0]
        correction = (delta.reshape(1, 48) @ self.residual.matrix).reshape(40, 3).numpy()
        poses = compose_trajectory(base, correction)
        if poses.shape != (40, 3) or not np.isfinite(poses).all():
            raise RuntimeError('Invalid predicted trajectory')
        return poses
