# Prompt Fungineer v2

**Same size model. 3 years of progress. How much better can we make it?**

In 2023, I fine-tuned GPT-2 (355M) to expand simple image prompts into detailed ones. In 2026, I retrained the concept using Qwen3-0.6B (596M) — same size class — on 9,400 Claude-generated prompt pairs. Then I trained the same architecture from scratch to measure how much pretraining actually contributes.

## The Experiment

Three models, one task: expand "sunset over the ocean" into a production-ready image generation prompt.

| Model | Params | Architecture | Weights | Training Data | Eval Loss |
|-------|--------|-------------|---------|---------------|-----------|
| **v1** (GPT-2) | 355M | 2019 | Pretrained | Web-scraped | unknown |
| **v2** (Qwen3) | 596M | 2025 | Pretrained | 9,400 Claude pairs | **2.67** |
| **v2-scratch** | 596M | 2025 | Random init | 9,400 Claude pairs | **6.02** |

## Results

### v1 vs v2: Architecture + Data Quality

v2 generates **27% more words** on average (117 vs 92), uses specific details (cameras, locations, lighting conditions), and gets the scene right where v1 hallucinates. We generated images from all prompts using Grok Imagine to prove better prompts → better images:

#### "sunset over the ocean"

| Basic Input | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|:-----------:|:---------------:|:----------------:|
| ![basic](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/sunset_over_the_ocean_basic.jpg) | ![v1](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/sunset_over_the_ocean_v1.jpg) | ![v2](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/sunset_over_the_ocean_v2.jpg) |
| *"sunset over the ocean"* | *"A breathtaking image of a tranquil, calm image of the horizon at sunset..."* | *"The Pacific Ocean at sunset, the sea surface in perfect mirror..."* |

#### "cherry blossoms in rain"

| Basic Input | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|:-----------:|:---------------:|:----------------:|
| ![basic](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/cherry_blossoms_in_rain_basic.jpg) | ![v1](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/cherry_blossoms_in_rain_v1.jpg) | ![v2](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/cherry_blossoms_in_rain_v2.jpg) |
| *"cherry blossoms in rain"* | *"...in a lush, tropical rainforest"* (wrong!) | *"...photographed at the moment of rain, petals floating. Fuji X-T5, 24mm f/4"* |

#### "wolf howling at the moon"

| Basic Input | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|:-----------:|:---------------:|:----------------:|
| ![basic](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/wolf_howling_at_the_moon_basic.jpg) | ![v1](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/wolf_howling_at_the_moon_v1.jpg) | ![v2](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/wolf_howling_at_the_moon_v2.jpg) |
| *"wolf howling at the moon"* | *"A haunting image of a deer..."* (hallucinated deer!) | *"A grey wolf stands in the moonlight, amber eyes locked on the full moon"* |

#### "neon-lit street at night"

| Basic Input | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|:-----------:|:---------------:|:----------------:|
| ![basic](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/neon-lit_street_at_night_basic.jpg) | ![v1](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/neon-lit_street_at_night_v1.jpg) | ![v2](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/neon-lit_street_at_night_v2.jpg) |
| *"neon-lit street at night"* | *"A captivating image of a bustling, neon-lit street..."* | *"A city street at 2am, every surface lit by a different coloured neon sign..."* |

#### "rainy tokyo street"

| Basic Input | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|:-----------:|:---------------:|:----------------:|
| ![basic](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/rainy_tokyo_street_basic.jpg) | ![v1](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/rainy_tokyo_street_v1.jpg) | ![v2](https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images/rainy_tokyo_street_v2.jpg) |
| *"rainy tokyo street"* | *"A vibrant and dramatic image of a bustling tokyo street..."* | *"A narrow alley in Shibuya during the rain, grey asphalt reflecting..."* |

### Pretrained vs From Scratch: The Pretraining Tax

The from-scratch model proves pretraining is the dominant factor:

```
Pretrained:    loss 3.5 → 2.67 (converged in ~100 steps)
From scratch:  loss 11.89 → 6.02 (still far from useful after 600 steps)
```

After 600 steps of training, the scratch model hadn't even reached where the pretrained model *started*. 9,400 samples is enough to teach a task but nowhere near enough to teach a language.

### The Quality Stack

| Factor | Contribution | Evidence |
|--------|-------------|---------|
| **Pretraining** | ~60% | Scratch eval loss 6.02 vs pretrained 2.67 |
| **Data quality** | ~25% | Claude-generated > web-scraped, same arch |
| **Architecture** | ~15% | Modern arch provides capacity but not knowledge |

## Training on Apple Silicon

Full fine-tuning Qwen3-0.6B on M4 Pro (64GB) required significant workarounds:

- **float32 only** — BF16 training hangs on MPS
- **batch_size=1** + gradient accumulation (OOM at batch=4)
- **Gradient checkpointing** — reduces activation memory 70%
- **HuggingFace Trainer** — raw PyTorch loops hang on MPS
- **~3 hours** per training run

See [WRITEUP.md](WRITEUP.md) for detailed logs of all 6 issues we hit and their fixes.

## Quick Start

```bash
git clone https://github.com/treadon/prompt-fungineer-v2
cd prompt-fungineer-v2

python3 -m venv venv && source venv/bin/activate
pip install torch transformers datasets wandb

# Train (pretrained base)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train_hf.py

# Train (from scratch — for comparison)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train_scratch.py

# Evaluate
python evaluate.py
```

## Links

- **v1 Model**: [treadon/prompt-fungineer-355M](https://huggingface.co/treadon/prompt-fungineer-355M)
- **v2 Model**: [treadon/prompt-fungineer-v2](https://huggingface.co/treadon/prompt-fungineer-v2)
- **Training Data**: [treadon/prompt-fungineer-v2-training-data](https://huggingface.co/datasets/treadon/prompt-fungineer-v2-training-data) (9,400 Claude-generated pairs)
- **W&B Dashboard**: [prompt-fungineer-v2](https://wandb.ai/actual-ritesh-org/prompt-fungineer-v2) (all training runs)
- **v1 Demo**: [HuggingFace Space](https://huggingface.co/spaces/treadon/prompt-fungineer-355M)
- **Full Writeup**: [WRITEUP.md](WRITEUP.md)
- **Experiment Design**: [EXPERIMENT.md](EXPERIMENT.md)

## Built With

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — Base model
- [Claude Code](https://claude.ai/claude-code) — Training data generation + development
- [W&B](https://wandb.ai) — Experiment tracking

*Built on a MacBook Pro M4 Pro. No cloud GPUs were used.*
