# VITAL single-file checkpoint — third-party notices

This model-only file excludes the public visual backbone and WAN VAE weights.
Submission author identities and original training metadata are not included.

- The public SimWAM checkpoint supplies the WAN-derived visual backbone and is
  downloaded separately. Source: https://huggingface.co/H-EmbodVis/SimWAM
- The public Wan2.2 VAE is downloaded separately.
  Source: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Action policy/F0 adapters, trajectory decoder, and residual policy are
  packaged as inference-only VITAL state dictionaries. Normalization and the
  constant prompt embedding are model buffers, not dataset records. The small
  proprioception encoder initialized from SimWAM is also retained.

The checkpoint's `notices` field includes this text and the upstream source
license texts. Public model weights retain their own model-card terms; source
code licenses do not replace model usage terms. Preserve attribution when
redistributing this file.
