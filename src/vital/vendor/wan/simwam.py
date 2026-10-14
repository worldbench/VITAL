from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ..logging import get_logger
from .action_dit import ActionDiT
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
logger = get_logger(__name__)

class SimWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(self, video_expert, action_expert: ActionDiT, mot: MoT, vae, text_encoder=None, tokenizer=None, text_dim: Optional[int]=None, proprio_dim: Optional[int]=None, device: str='cpu', torch_dtype: torch.dtype=torch.float32, video_train_shift: float=5.0, video_infer_shift: float=5.0, video_num_train_timesteps: int=1000, action_train_shift: float=5.0, action_infer_shift: float=5.0, action_num_train_timesteps: int=1000, action_prediction_type: str='velocity', action_sigma_clamp_min: float=0.1, loss_lambda_video: float=1.0, loss_lambda_action: float=1.0):
        super().__init__()
        if action_prediction_type not in ('velocity', 'sample'):
            raise ValueError(f"`action_prediction_type` must be 'velocity' or 'sample', got {action_prediction_type!r}")
        if not 0.0 < action_sigma_clamp_min <= 1.0:
            raise ValueError(f'`action_sigma_clamp_min` must be in (0, 1], got {action_sigma_clamp_min}')
        self.action_prediction_type = str(action_prediction_type)
        self.action_sigma_clamp_min = float(action_sigma_clamp_min)
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError('`text_dim` is required when `text_encoder` is not loaded.')
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=video_num_train_timesteps, shift=video_train_shift)
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=action_num_train_timesteps, shift=action_train_shift)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift)
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.to(self.device)

    def _action_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """σ = t/T (SimWAM noising convention), clamped to [sigma_min, 1]."""
        sigma = timestep.to(dtype=torch.float32) / float(self.train_action_scheduler.num_train_timesteps)
        return sigma.clamp(min=self.action_sigma_clamp_min, max=1.0)

    def _action_x_to_v(self, pred_x: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Convert a clean-sample prediction x̂ into a flow-matching velocity v̂.

        With x_σ = (1-σ)·x_data + σ·noise, v* = noise - x_data, and x̂ = x̂_data:
            v̂ = (x_σ - x̂) / σ
        σ is clamped via `_action_sigma` for numerical stability near σ→0.
        """
        sigma = self._action_sigma(timestep).to(device=noisy_action.device)
        sigma = sigma.view(-1, *[1] * (noisy_action.ndim - 1))
        return (noisy_action - pred_x) / sigma.to(dtype=noisy_action.dtype)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def _append_proprio_to_context(self, context: torch.Tensor, context_mask: torch.Tensor, proprio: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return (context, context_mask)
        if proprio.ndim != 2:
            raise ValueError(f'`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}')
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f'`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}')
        proprio_token = self.proprio_encoder(proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1))

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(video_tensor, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f'`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}')
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    @torch.no_grad()
    def _build_mot_attention_mask(self, video_seq_len: int, action_seq_len: int, video_tokens_per_frame: int, device: torch.device) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(video_seq_len=video_seq_len, video_tokens_per_frame=video_tokens_per_frame, device=device)
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _predict_action_noise_with_cache(self, latents_action: torch.Tensor, timestep_action: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor, video_kv_cache: list[dict[str, torch.Tensor]], attention_mask: torch.Tensor, video_seq_len: int) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(action_tokens=latents_action, timestep=timestep_action, context=context, context_mask=context_mask)
        action_tokens = self.mot.forward_action_with_video_cache(action_tokens=action_pre['tokens'], action_freqs=action_pre['freqs'], action_t_mod=action_pre['t_mod'], action_context_payload={'context': action_pre['context'], 'mask': action_pre['context_mask']}, video_kv_cache=video_kv_cache, attention_mask=attention_mask, video_seq_len=video_seq_len)
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action(self, prompt: Optional[str], input_image: torch.Tensor, action_horizon: int, proprio: Optional[torch.Tensor]=None, context: Optional[torch.Tensor]=None, context_mask: Optional[torch.Tensor]=None, negative_prompt: Optional[str]=None, text_cfg_scale: float=1.0, num_inference_steps: int=20, sigma_shift: Optional[float]=None, seed: Optional[int]=None, rand_device: str='cpu', tiled: bool=False) -> dict[str, Any]:
        """
        Infer action sequence from a single image and optional prompt/context.
        """
        self.eval()
        if str(getattr(self.video_expert, 'video_attention_mask_mode', '')) != 'first_frame_causal':
            raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'`.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f'`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}')
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f'`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})')
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError('`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.')
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f'`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}')
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f'`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}')
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn((1, action_horizon, self.action_expert.action_dim), generator=generator, device=rand_device, dtype=torch.float32).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError('`prompt` and `context/context_mask` are mutually exclusive.')
        if not use_prompt and (not use_context):
            raise ValueError('Either `prompt` or both `context/context_mask` must be provided.')
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError('`context` and `context_mask` must be both provided together.')
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(f'`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}')
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context=context, context_mask=context_mask, proprio=proprio)
        timestep_video = torch.zeros((first_frame_latents.shape[0],), dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(x=first_frame_latents, timestep=timestep_video, context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag)
        video_seq_len = int(video_pre['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(video_seq_len=video_seq_len, action_seq_len=latents_action.shape[1], video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']), device=video_pre['tokens'].device)
        video_kv_cache = self.mot.prefill_video_cache(video_tokens=video_pre['tokens'], video_freqs=video_pre['freqs'], video_t_mod=video_pre['t_mod'], video_context_payload={'context': video_pre['context'], 'mask': video_pre['context_mask']}, video_attention_mask=attention_mask[:video_seq_len, :video_seq_len])
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps, device=self.device, dtype=latents_action.dtype, shift_override=sigma_shift)
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action_posi = self._predict_action_noise_with_cache(latents_action=latents_action, timestep_action=timestep_action, context=context, context_mask=context_mask, video_kv_cache=video_kv_cache, attention_mask=attention_mask, video_seq_len=video_seq_len)
            pred_action = pred_action_posi
            if self.action_prediction_type == 'sample':
                pred_action = self._action_x_to_v(pred_action, latents_action, timestep_action)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        return {'action': latents_action[0].detach().to(device='cpu', dtype=torch.float32)}

