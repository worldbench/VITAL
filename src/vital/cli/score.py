"""Score freshly inferred trajectories with the official NAVSIM v1 simulator."""
import argparse
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/navtest_scores.csv'))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from vital.services.scoring import score_predictions
    score_predictions(**vars(args))


if __name__ == "__main__":
    main()
