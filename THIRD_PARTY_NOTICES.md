# Third-party components

VITAL uses the following public third-party components and retains their
source-code licenses and model attribution.

- `src/vital/vendor/wan/` contains inference primitives adapted from
  [SimWAM](https://github.com/H-EmbodVis/SimWAM), with the MIT notice preserved
  in `THIRD_PARTY_LICENSES/SimWAM.txt`. Training entry points, remote model
  loaders, and workspace-specific dependencies are not included.
- The frozen image VAE uses [Wan2.2](https://github.com/Wan-Video/Wan2.2);
  its Apache 2.0 text is preserved in `THIRD_PARTY_LICENSES/Wan2.2.txt`.
  Upstream code/weights retain their applicable licenses. The visual backbone
  comes from the pinned public SimWAM base weights. Both the backbone and VAE
  are downloaded by users, not redistributed in `vital.pt`. The selected
  action/F0 adaptation, proprioception encoder, decoder, and VITAL residual
  policy are supplied in that single file. The public model-card terms apply
  separately from upstream source-code licenses.
- The official NAVSIM navtest split configuration is copied unchanged from
  NAVSIM commit `0811876c274e8b058ab2be9b3dcd4d37bd23f177`. Its license is
  retained in `THIRD_PARTY_LICENSES/NAVSIM.txt`.
- NAVSIM and nuPlan are installed as public dependencies for metric-cache
  construction and scoring. Dataset downloads remain subject to their
  providers' terms, separately from any code license.

Source adaptation includes namespacing imports under `vital`, removing
unused training/loading entry points, and keeping the numerical inference
path and official scoring definitions.
