# User-managed data

No dataset, inference samples, feature/text caches, metric caches or score
exports are supplied by the authors. Download official NAVSIM test metadata,
camera blobs and nuPlan maps using the [README instructions](../README.md#3-download-public-navsim-data).

Expected layout:

```text
data/navsim/navsim_logs/test/<log>.pkl
data/navsim/sensor_blobs/test/<log>/CAM_F0/<image>.jpg
data/navsim/maps/...
```

The model reads raw images and ego/navigation fields itself.
Run `vital-prepare-metrics` to create your own official metric caches.
Do not commit downloaded data or generated caches.
