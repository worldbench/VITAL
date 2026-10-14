#!/usr/bin/env python3
"""Maintainer utility: export one VITAL inference file, excluding public base models.

Use --legacy-dir to merge a previous inference-only release, or pass the six
trusted original inputs. Training metadata is never copied to the output.
"""
import argparse
import json
from pathlib import Path
import torch
import yaml
from vital.weights import FORMAT, UPSTREAM, load_bundle, sha256


def read(path):
    # Original checkpoints are maintainer-owned trusted files only.
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def tensors(state):
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected only state-dictionary tensors')
    # Clone to avoid serializing unrelated tensors sharing a large flat storage.
    return {key: value.detach().cpu().clone().contiguous() for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-dir', type=Path)
    source_names = ('base', 'adapter', 'decoder', 'residual', 'context', 'model-config')
    for name in source_names:
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/vital.pt'))
    parser.add_argument('--manifest', type=Path, default=Path('src/vital/resources/weights_manifest.json'))
    args = parser.parse_args()
    originals = [getattr(args, name.replace('-', '_')) for name in source_names]
    if args.legacy_dir and any(originals):
        parser.error('Use --legacy-dir OR the six original inputs, not both')
    if not args.legacy_dir and not all(originals):
        parser.error('Provide --legacy-dir or all six original inputs')
    if args.output.exists() or args.manifest.exists():
        parser.error('Output or manifest exists; choose new paths')

    if args.legacy_dir:
        def legacy(name):
            return torch.load(args.legacy_dir / name, weights_only=True, map_location='cpu', mmap=True)
        config = json.loads((args.legacy_dir / 'model_config.json').read_text())
        core = legacy('action_policy.pt')
        decoder = legacy('trajectory_decoder.pt')
        residual = legacy('residual_policy.pt')
    else:
        base, adapter = read(args.base), read(args.adapter)
        context = read(args.context)
        fixed, mask = context['context'].to(torch.bfloat16).clone(), context['mask'].bool()
        if fixed.shape != (256, 4096) or mask.shape != (256,):
            raise ValueError('Wrong constant-prompt tensor shape')
        fixed[~mask] = 0
        core = {'action_expert': tensors(adapter['action_expert']),
                'f0_adapter': tensors(adapter['f0_adapter']),
                'proprio_encoder': tensors(base['proprio_encoder']),
                'action_normalizer': tensors(adapter['action_normalizer']),
                'static_context': fixed, 'static_context_mask': torch.ones_like(mask)}
        decoder = tensors(read(args.decoder)['model'])
        source = read(args.residual)
        residual = {'state_dict': tensors(source['state_dict']), 'config': {
            key: source['actor_config'][key] for key in
            ('kind', 'exploration_rms', 'max_xy', 'max_yaw', 'max_rms_q')}}
        model = yaml.safe_load(args.model_config.read_text())
        config = {'video': model['video_dit_config'], 'action': model['action_dit_config'],
                  'f0_adapter': {key: adapter['f0_adapter_config'][key] for key in
                                ('last_n_blocks', 'rank', 'alpha', 'dropout', 'target_modules')}}
        for kind in ('video', 'action'):
            config[kind]['use_gradient_checkpointing'] = False

    release = Path(__file__).resolve().parents[1]
    notices = {'THIRD_PARTY_NOTICES.md': (release / 'docs/WEIGHT_NOTICES.md').read_text()}
    for name in ('SimWAM', 'Wan2.2'):
        notices[f'LICENSE_{name}.txt'] = (release / 'THIRD_PARTY_LICENSES' / f'{name}.txt').read_text()
    bundle = {'format': FORMAT, 'nfe': 1, 'seed': 0, 'model_config': config,
              'action_policy': core, 'trajectory_decoder': decoder, 'residual_policy': residual,
              'upstream': UPSTREAM, 'notices': notices}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save(bundle, stream)
    load_bundle(args.output)
    manifest = {'format': FORMAT, 'filename': 'vital.pt', 'bytes': args.output.stat().st_size,
                'sha256': sha256(args.output)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    print(f'Ready for upload: {args.output} ({manifest["bytes"] / 1e9:.3f} GB)', flush=True)


if __name__ == '__main__':
    main()
