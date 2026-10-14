"""Build NAVSIM metric caches from locally downloaded logs and maps."""
import os
from pathlib import Path

from vital.data.navsim import scenes


def prepare_metrics(*, data_root, cache, limit=0, tokens=None):
    data_root, cache = Path(data_root), Path(cache)
    tokens = Path(tokens) if tokens is not None else None
    root = data_root.resolve()
    os.environ['NUPLAN_MAPS_ROOT'] = str(root / 'maps')
    os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    if limit < 0 or not (root / 'maps').is_dir():
        raise ValueError('Download maps under DATA_ROOT/maps and use a nonnegative limit')
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    processor = MetricCacheProcessor(str(cache), force_feature_computation=False)
    selected = tokens.read_text().splitlines() if tokens else None
    count = 0
    for token, frames in scenes(root, selected, limit):
        scene = Scene.from_scene_dict_list(frames, None, num_history_frames=4,
                                           num_future_frames=10, sensor_config=SensorConfig.build_no_sensors())
        scenario = NavSimScenario(scene, map_root=str(root / 'maps'), map_version='nuplan-maps-v1.0')
        result = processor.compute_metric_cache(scenario)
        if result is None:
            raise RuntimeError(f'Cache failed for {token}')
        count += 1
        if count == 1 or count % 100 == 0:
            print(f'Cached {count} scenes', flush=True)
    print(f'Prepared {count} official metric caches in {cache}', flush=True)

