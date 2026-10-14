"""ADE/FDE helpers for the 10 Hz prediction versus 2 Hz recording protocol."""

from __future__ import annotations

import numpy as np


def _interp_pose(poses: np.ndarray, source_dt: float, target_times: np.ndarray) -> np.ndarray:
    source_times = np.arange(1, len(poses) + 1, dtype=np.float64) * float(source_dt)
    times = np.concatenate(([0.0], source_times))
    values = np.vstack([np.zeros((1, 3), dtype=np.float64), poses[:, :3]])
    yaw = np.unwrap(values[:, 2])
    return np.column_stack(
        [np.interp(target_times, times, values[:, 0]),
         np.interp(target_times, times, values[:, 1]),
         np.interp(target_times, times, yaw)]
    )


def trajectory_metrics(predicted: np.ndarray, recorded: np.ndarray, *, predicted_dt: float = 0.1,
                       recorded_dt: float = 0.5) -> dict[str, float]:
    """Return position ADE/FDE in metres after interpolation to GT timestamps."""

    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(recorded, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] < 2 or gt.shape[1] < 2:
        raise ValueError("predicted and recorded must be [T, >=2]")
    if len(pred) == 0 or len(gt) == 0:
        raise ValueError("empty trajectory")
    target_times = np.arange(1, len(gt) + 1, dtype=np.float64) * float(recorded_dt)
    pred_at_recorded = _interp_pose(pred, predicted_dt, target_times)
    distances = np.linalg.norm(pred_at_recorded[:, :2] - gt[:, :2], axis=1)
    return {"ADE_m": float(distances.mean()), "FDE_m": float(distances[-1]), "num_steps": int(len(gt))}


def assert_one_step(num_inference_steps: int) -> None:
    if int(num_inference_steps) != 1:
        raise ValueError("This public entry point is the NFE=1 protocol only.")
