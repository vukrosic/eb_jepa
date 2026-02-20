# Fast Experiment Guide for Image JEPA (RTX 4090)

This guide details how to perform rapid, compute-efficient training runs for the Image JEPA (VICReg) model on an RTX 4090 (or similar 24GB VRAM GPU). 

By carefully balancing GPU architecture constraints against the mathematical requirements of self-supervised learning, this setup enables you to test hypotheses and achieve **~55% linear probe accuracy in under 12 minutes**.

---

## 🚀 The Rapid Config (`rapid.yaml`)

The fundamental bottleneck on high-end GPUs like the RTX 4090 is often **memory bandwidth**, not pure compute. Pushing the batch size to the VRAM limit creates a memory transfer bottleneck. 

To achieve maximum throughput (~3,800 images/sec) and rapid feature decorrelation, use the following configuration (`examples/image_jepa/cfgs/rapid.yaml`):

```yaml
meta:
  seed: 42
  device: auto

data:
  dataset: cifar10
  batch_size: 3072      # Optimal for RTX 4090 found in benchmarking
  num_workers: 16       # Maximizes CPU-to-GPU memory transfer

model:
  type: resnet
  use_projector: true
  proj_hidden_dim: 4096 # Wide projector = faster feature decorrelation
  proj_output_dim: 4096 

loss:
  type: vicreg
  std_coeff: 25.0
  cov_coeff: 1.0

optim:
  epochs: 50            # Takes ~11-12 minutes total
  lr: 0.707             # Best LR for rapid aggressive representation learning
  weight_decay: 1.0e-4
  warmup_epochs: 5      # Short warmup to stabilize the aggressive LR
  warmup_start_lr: 3.0e-5
  min_lr: 0.0

logging:
  log_wandb: false
  log_every: 5
  save_every: 50
  tqdm_silent: false

training:
  use_amp: true
  dtype: bfloat16       # Hardware accelerated precision on Ada Lovelace architecture
```

---

## 🏃 Execution

To run the rapid experiment:

```bash
# Run command
EBJEPA_DSETS="./datasets" python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/rapid.yaml
```

## 📊 Summary of Benchmarks (RTX 4090)

### ResNet-18 Efficiency
| Epochs | Time | Val Acc% | Note |
| :--- | :--- | :--- | :--- |
| 10 | 2.5m | 31.6% | Early signal |
| 50 | 11.5m | **55.2%** | **Standard Rapid Run** |
| 800 | 3.2h | 82.1% | SOTA (requires `bs: 512`) |

### ViT-S Ablation (30 Epochs)
- **Best LR**: **0.02** leads to **33.12%**.
- **Issue**: LR > 0.1 causes instability/NaNs.
- **Warmup**: Reducing warmup below 10 epochs causes a ~6% accuracy drop.

## � Quick Tips:
1. **Bandwidth**: If throughput is low, increase `--data.num_workers=16`.
2. **Stability**: If loss spikes, lower LR by 2x or double the warmup.
3. **Precision**: Always use `dtype: bfloat16` on RTX 40/H100 series for a 2x speedup over `float32`.