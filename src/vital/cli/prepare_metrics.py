"""Build official NAVSIM v1 metric caches locally from downloaded logs/maps."""
import argparse
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path('data/metric_cache'))
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--tokens', type=Path)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from vital.services.metric_cache import prepare_metrics
    prepare_metrics(**vars(args))


if __name__ == "__main__":
    main()
