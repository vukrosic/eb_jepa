<h1 align="center">
    <p>⚡ <b>EB-JEPA</b></p>
</h1>

<h2 align="center">
    <p><i>Energy-Based Joint-Embedding Predictive Architectures</i></p>
</h2>

<div align="center">
  <a href="https://github.com/facebookresearch/eb_jepa"><img src="https://img.shields.io/badge/Github-facebookresearch/eb--jepa-black?logo=github"/></a>
  <a href="https://arxiv.org/abs/2602.03604"><img src="https://img.shields.io/badge/arXiv-2602.03604-b5212f?logo=arxiv"/></a>
</div>

<p align="center">
  <b><a href="https://ai.facebook.com/research/">Meta AI Research, FAIR</a></b>
</p>

An open source library for learning representations using joint embedding predictive architectures.

<p align="center">
  <img src="docs/archi-schema-eb-jepa.png" alt="EB-JEPA Architecture" width="600">
</p>

---

## 📚 Examples

- **[Image JEPA](examples/image_jepa/README.md)**: SSL on CIFAR-10 with ViT/ResNet.
- **[Video JEPA](examples/video_jepa/README.md)**: Predicting next image representations.
- **[AC Video JEPA](examples/ac_video_jepa/README.md)**: World modeling and planning.

---

## 🚀 Quick Start

### Installation
We recommend using [uv](https://docs.astral.sh/uv/):
```bash
uv sync
source .venv/bin/activate
```

### Training
```bash
# Run Image SSL with optimized ViT configuration
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/rapid_vit.yaml
```

**⚡ Optimized ViT Config:** Our default `rapid_vit.yaml` uses **Patch Size 4** and **LR 0.005**, achieving **33.1%** accuracy (vs 29.5% baseline) by better utilizing spatial information.

---

## 👩‍💻 Development

```bash
# Test
uv run pytest tests/

# Format
python -m isort eb_jepa examples tests
python -m black eb_jepa examples tests
```

---

## 📚 Citation

```bibtex
@misc{terver2026lightweightlibraryenergybasedjointembedding,
      title={A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures},
      author={Basile Terver others},
      year={2026},
      eprint={2602.03604},
      url={https://arxiv.org/abs/2602.03604}
}
```

## 📄 License
Apache 2.0. See [LICENSE](LICENSE.md).
