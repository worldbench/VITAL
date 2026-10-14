import numpy as np
import torch

def compose_trajectory(base, correction):
    """Shared train/deploy arithmetic: T_IL + the LINEAR residual increment.

    This is algebraically D_v(z_v)+D_r(z_r+dq). The PCA affine mean is already
    in T_IL, so it MUST NOT be added a second time. Zero delta returns exactly
    the original FP32 trajectory, including already wrapped yaw bits.
    """
    base = np.asarray(base, dtype=np.float32)
    correction = np.asarray(correction, dtype=np.float32)
    result = base + correction
    result[..., 2] = np.arctan2(np.sin(result[..., 2]), np.cos(result[..., 2]))
    zero = (correction == 0).all(axis=(-2, -1))
    return np.where(zero[..., None, None], base, result).astype(np.float32)


def physical_project(delta_q, matrix, max_xy=0.75, max_yaw=0.15, max_rms_q=4.0):
    pose = (delta_q @ matrix).reshape(*delta_q.shape[:-1], 40, 3)
    xy_ratio = pose[..., :2].norm(dim=-1).amax(dim=-1) / max_xy
    yaw_ratio = pose[..., 2].abs().amax(dim=-1) / max_yaw
    # vector_norm has a finite zero subgradient; sqrt(mean(q**2)) does not.
    q_ratio = torch.linalg.vector_norm(delta_q, dim=-1) / (delta_q.shape[-1]**.5 * max_rms_q)
    scale = torch.maximum(torch.maximum(xy_ratio, yaw_ratio), q_ratio).clamp_min(1)
    return delta_q / scale[..., None]


def build_features(normalized_action, trajectory, *, hidden, ego, spatial=None, kind="hidden"):
    arrays = [np.asarray(hidden, dtype=np.float32).reshape(-1),
              np.asarray(ego, dtype=np.float32).reshape(-1),
              np.asarray(normalized_action, dtype=np.float32).reshape(-1),
              np.asarray(trajectory, dtype=np.float32).reshape(-1)]
    if arrays[0].size != 4096 or arrays[1].size != 8 or arrays[2].size != 304 or arrays[3].size != 120:
        raise ValueError("Expected final action hidden[4,1024], ego[8], action[4,76], poses[40,3]")
    if kind == "hidden_spatial":
        space = np.asarray(spatial, dtype=np.float32).reshape(-1)
        if space.size != 8192:
            raise ValueError("Expected current F0 spatial feature[8192]")
        arrays.append(space)
    elif kind != "hidden":
        raise ValueError(kind)
    result = np.concatenate(arrays)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite deployment feature")
    return result

