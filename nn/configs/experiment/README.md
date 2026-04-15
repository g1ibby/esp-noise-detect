# Experiment Presets

Drop Hydra experiment bundles here. Example usage:

```yaml
# configs/experiment/pump_ablation.yaml
defaults:
  - override /augment: light
  - override /dataset: manifest
  - override /model: tinyconv
  - _self_

trainer:
  max_epochs: 40

dataset:
  window_s: 0.64
```

Run with:

```
uv run -m noise_detect.train experiment=pump_ablation
```
