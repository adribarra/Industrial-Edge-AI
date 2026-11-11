# Models

Contents are git-ignored. Layout of `models/patchcore_resnet18_screw/`:

- `resnet18-f37072fd.pth` - torchvision ResNet18 ImageNet weights (~47 MB),
  from https://download.pytorch.org/models/resnet18-f37072fd.pth
  (torchvision code: BSD-3; weights trained on ImageNet).
- `memory_bank.pt` - patch features built locally from the normal screw
  `train` images by `scripts/build_memory_bank.py` (see root README).

Create both with: `python scripts/build_memory_bank.py`
