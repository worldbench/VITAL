"""Raw NAVSIM logs -> real one-step model predictions."""
import argparse
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=Path('checkpoints/vital.pt'))
    p.add_argument('--simwam-weights', type=Path, default=Path('checkpoints/upstream/SimWAM.pt'),
                   help='Pinned public SimWAM checkpoint containing the trained WAN backbone')
    p.add_argument('--wan-vae', type=Path, default=Path('checkpoints/upstream/Wan2.2_VAE.pth'),
                   help='Original public Wan2.2 VAE checkpoint')
    p.add_argument('--output', type=Path, default=Path('outputs/navtest_nfe1.npz'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0, help='0 = complete navtest')
    p.add_argument('--tokens', type=Path, help='Optional local newline-separated token subset')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from vital.services.inference import run_inference
    run_inference(**vars(args))


if __name__ == "__main__":
    main()
