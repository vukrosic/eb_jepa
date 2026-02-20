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
  batch_size: 512       # 512 avoids the memory bandwidth bottleneck of the RTX 4090
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
  lr: 0.5               # High LR for rapid, aggressive clustering
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
# Point to your datasets directory (if downloaded)
export EBJEPA_DSETS="./datasets" 

# Execute the training script
python -m examples.image_jepa.main \
    --fname examples/image_jepa/cfgs/rapid.yaml \
    --meta.model_folder "checkpoints/image_jepa/rapid_run"
```

## 📊 Analysis and Expectations

### What to Expect
- **Training Time:** ~13-14 seconds per epoch.
- **Total Compute Time:** ~11.5 minutes for 50 epochs.
- **Loss:** Drops rapidly from `~4.8` to `< 0.7`
- **Validation Accuracy (Linear Probe):** Surpasses `50%` around epoch 30, finishing near `55%`.

### Why this works so well:
1.  **Wide Projector (`4096` dim)**: The massive multi-dimensional projection space allows the VICReg Covariance loss to decorrelate features much faster in early epochs.
2.  **Aggressive Learning Rate (`0.5`)**: Paired with LARS optimization, hitting the model hard early forces rapid clustering of representations.
3.  **Optimal Batch Sizing (`512`)**: Keeps the compute units saturated while sitting perfectly in the GDDR6X memory bandwidth pipeline.

---

### Need >80% Accuracy?
If your rapid experiments are successful and you want a full State-of-the-Art representation, you must train for much longer. Use `batch_size: 512`, `lr: 0.3`, and set `epochs: 800`. This will take roughly **3 hours** on an RTX 4090.
