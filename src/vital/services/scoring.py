"""Evaluate predictions with the official NAVSIM v1 scoring implementation."""
import csv
from dataclasses import asdict
import json
import lzma
import pickle
from pathlib import Path

import numpy as np


def score_predictions(*, predictions, cache, output):
    predictions, cache, output = Path(predictions), Path(cache), Path(output)
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    if output.exists():
        raise ValueError('Output already exists; choose a new output path')
    data = np.load(predictions, allow_pickle=False)
    tokens, poses = data['tokens'], data['poses']
    if len(tokens) == 0 or poses.shape != (len(tokens), 40, 3) or not np.isfinite(poses).all():
        raise ValueError('Expected nonempty finite [N,40,3] predictions')
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate tokens')
    cache_files = {}
    for path in cache.rglob('metric_cache.pkl'):
        token = path.parent.name
        if token in cache_files:
            raise ValueError(f'Duplicate metric cache for {token}')
        cache_files[token] = path
    missing = set(tokens) - set(cache_files)
    if missing:
        raise RuntimeError(f'Missing {len(missing)} metric caches; run vital-prepare-metrics first')
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    simulator = PDMSimulator(proposal_sampling=sampling)
    scorer = PDMScorer(proposal_sampling=sampling, config=PDMScorerConfig(
        progress_weight=5., ttc_weight=5., comfortable_weight=2., driving_direction_weight=0.))
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open('x', newline='') as handle:
        writer = None
        for token, prediction in zip(tokens, poses, strict=True):
            # Caches are generated locally by the trusted official scorer.
            with lzma.open(cache_files[str(token)], 'rb') as source:
                cache = pickle.load(source)
            result = pdm_score(cache, Trajectory(prediction, sampling), sampling, simulator, scorer)
            row = dict(token=str(token), valid=True, **asdict(result))
            if not all(np.isfinite(v) for k, v in row.items() if k != 'token'):
                raise RuntimeError(f'Nonfinite score for {token}')
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); handle.flush(); rows.append(row)
    summary = dict(protocol='NAVSIM v1 PDMS', scenes=len(rows),
                   pdms=100 * float(np.mean([r['score'] for r in rows])),
                   scope='full navtest' if len(rows) == 12146 else 'subset; not paper reproduction')
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

