# Datasets

MVTec AD, `screw` category only (about 195 MB compressed, 188 MB extracted).

    python scripts/download_dataset.py        # -> datasets/mvtec/screw/

Contents of this folder are git-ignored. Do not commit dataset files.

- Source: https://www.mvtec.com/company/research/datasets/mvtec-ad
- License: CC BY-NC-SA 4.0 (non-commercial; attribution; share-alike)
- Citation: Bergmann et al., "MVTec AD - A Comprehensive Real-World Dataset for
  Unsupervised Anomaly Detection", CVPR 2019.

Screw split: train 320 (normal only), test 160 (41 good, 119 anomalous with masks).
