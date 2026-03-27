---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - prompt-engineering
  - image-generation
  - stable-diffusion
  - midjourney
  - dall-e
  - flux
  - distillation
  - qwen3
base_model: Qwen/Qwen3-0.6B
datasets:
  - treadon/prompt-fungineer-v2-training-data
---

# Prompt Fungineer v2

**Takes a simple 2-10 word image description and expands it into a detailed 50-150 word prompt for AI image generators.**

Prompt Fungineer v2 is a full fine-tune of [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (596M parameters) trained on 9,400 Claude-generated prompt pairs. It produces detailed, richly specific prompts suitable for Midjourney, Stable Diffusion, DALL-E, Flux, and other image generation models.

This is the successor to [treadon/prompt-fungineer-355M](https://huggingface.co/treadon/prompt-fungineer-355M), which was built on GPT-2 Medium in 2023.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "treadon/prompt-fungineer-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

def expand_prompt(short_description: str, max_new_tokens: int = 256) -> str:
    messages = [
        {
            "role": "user",
            "content": f"Expand this into a detailed image generation prompt: {short_description}"
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

# Example
import torch
print(expand_prompt("cherry blossoms in rain"))
```

**Example output:**

> A cherry blossom tree in full bloom photographed at the moment of rain, the petals floating and falling in the downpour. Fuji X-T5, 24mm f/4 ultra-wide lens, f/8, ISO 800, 1/1000s. The petals are translucent pink and white, each one catching the raindrop light differently. The rain falls in the background as a soft white blur. Color: vivid pink blossom against white rain, the sky behind in soft grey. Matte finish. The delicate pink against the wet grey sky. Japanese seasonal beauty.

## Model Details

| Property | Value |
|---|---|
| **Base model** | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (2025 architecture) |
| **Parameters** | 596M |
| **Training method** | Full fine-tune (all parameters unfrozen) |
| **Training data** | 9,400 Claude-generated prompt pairs |
| **Categories** | 11 (landscapes, portraits, sci-fi, abstract, architecture, animals, historical, food, macro, emotions, vehicles) |
| **Style distribution** | 5 styles (photographic 52%, cinematic 14%, painterly 14%, digital art 10%, illustration 9%) |
| **Epochs** | 4 |
| **Learning rate** | 2e-5, cosine schedule with 6% warmup |
| **Effective batch size** | 32 (batch 1 x gradient accumulation 32) |
| **Precision** | float32 |
| **Gradient checkpointing** | Enabled |
| **Max sequence length** | 384 tokens |
| **Training time** | ~2h 50m on Apple M4 Pro (MPS backend) |
| **Final train loss** | 1.51 |
| **Final eval loss** | 2.67 |
| **Optimizer** | AdamW (weight decay 0.01) |
| **License** | Apache 2.0 |

## v1 vs v2 Comparison

Prompt Fungineer v2 generates longer, more specific, and more technically detailed prompts compared to v1.

| Metric | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|---|---|---|
| **Parameters** | 355M | 596M (1.7x) |
| **Architecture** | GPT-2 (2019) | Qwen3 (2025) |
| **Avg output length** | 92 words | 117 words (+27%) |
| **Avg generation time** | 2.6s | 4.8s |
| **Training data** | Web-scraped prompts | 9,400 Claude-generated pairs |

### Side-by-Side Examples

| Input | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|---|---|---|
| **"sunset over the ocean"** | Generic: "A breathtaking image of a tranquil, calm image of the horizon at sunset, revealing a serene, crystal-clear ocean..." | Specific: "The Pacific Ocean at sunset, the sea surface in perfect mirror -- the sky above reflected in the dark water below, the sky and its reflection creating a continuous white and gold spectacle from horizon to zenith." |
| **"neon-lit street at night"** | Generic: "A captivating image of a bustling, neon-lit street at night, with pedestrians weaving through..." | Specific: "A city street at 2am, every surface lit by a different coloured neon sign -- red traffic lights, blue bus stop signs, green bus number boards, yellow taxi stop boards." |
| **"cherry blossoms in rain"** | Wrong scene: confused with "a lush, tropical rainforest" | Correct scene: cherry blossom tree in rain, named specific camera (Fuji X-T5, 24mm f/4), technical settings (f/8, ISO 800, 1/1000s) |
| **"wolf howling at the moon"** | Hallucinated: "A haunting image of a deer stalking the night sky" | Correct subject: "A grey wolf stands in the moonlight, its amber eyes locked on the full moon. The moon is a perfect silver disc above the horizon." |
| **"frozen waterfall"** | Vague: "An enchanting image of a majestic waterfall cascading from a cliff" (missed the "frozen" part) | Accurate: "A waterfall frozen at its peak, the water solid and translucent in the cold air, the waterfall walls visible through the ice." |
| **"grandmother baking bread"** | Odd details: "leaving a trail of pink and orange bread crumbs on the cob" | Grounded: "A grandmother in her 80s holds a large wooden mixing board covered in a sheet of sourdough, steam rising from the mixture. Her hands are flour-dusted." |

### Key Improvements in v2

- **Specificity:** v2 names concrete details (locations, camera models, times of day) instead of generic adjectives
- **Technical vocabulary:** v2 includes camera specs (f-stops, ISO, lens mm), composition directions, and lighting terms
- **Subject accuracy:** v2 stays true to the input description; v1 sometimes drifted to unrelated subjects
- **Scene coherence:** v2 builds a unified scene with a consistent mood rather than stacking unrelated descriptors

## Known Limitations

- **Chat template artifacts:** v2 occasionally includes fragments of the chat template (e.g., `user`, `assistant` markers) in the generated output. Post-processing to strip these is recommended.
- **Hallucinated details:** v2 can introduce incorrect details. For example, "old man playing chess" produced a prompt mentioning a "mahjong table" despite correctly describing chess pieces and a chessboard.
- **Slower generation:** v2 averages 4.8s per prompt vs 2.6s for v1, due to the larger model and longer outputs.
- **Repetitive phrasing:** v2 sometimes falls into repetitive philosophical loops at the end of longer outputs (e.g., "The moon that is watching and the animal that is watching the moon. The moon that is always there.").
- **Overfitting gap:** The gap between train loss (1.51) and eval loss (2.67) suggests some overfitting, which may contribute to occasional repetition or formulaic outputs.

## Training Data

The model was trained on [treadon/prompt-fungineer-v2-training-data](https://huggingface.co/datasets/treadon/prompt-fungineer-v2-training-data), a dataset of 9,400 prompt pairs generated by Claude (Sonnet/Opus).

**Category distribution:**

| Category | Count |
|---|---|
| landscapes_nature | 1,288 |
| portraits_people | 1,091 |
| architecture_urban | 979 |
| historical_vintage | 940 |
| emotions_concepts | 812 |
| abstract_artistic | 774 |
| food_stilllife | 750 |
| macro_closeup | 742 |
| animals_wildlife | 701 |
| scifi_fantasy | 681 |
| vehicles_technology | 642 |

Each pair consists of a short input (1-9 words, avg 4.8) and a detailed output (53-181 words, avg 90.5). The data spans 5 style profiles: photographic, cinematic, painterly, digital art, and illustration.

Quality controls applied during generation:
- No copyrighted names or characters
- No NSFW content
- Varied sentence structures
- Prompt lengths between 50-150 words
- Each pair validated: input at most 15 words, output at least 20 words

## Training Infrastructure

Trained on an Apple M4 Pro with 64GB unified memory using the MPS backend. Key lessons from training on Apple Silicon:

- **Use HuggingFace Trainer**, not raw PyTorch loops (MPS shader compilation can hang with manual device placement)
- **Use float32** (BF16 training on MPS causes silent hangs or incorrect results)
- **Use batch_size=1 + gradient accumulation** (float32 activations for 28-layer models blow up memory at larger batch sizes)
- **Enable gradient checkpointing** (trades ~30% more compute for ~70% less activation memory)

## Citation

```bibtex
@misc{prompt-fungineer-v2,
  author = {Ritesh Khanna},
  title = {Prompt Fungineer v2: Distilling Prompt Engineering Into a Small Local Model},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/treadon/prompt-fungineer-v2}
}
```

## Predecessor

- [treadon/prompt-fungineer-355M](https://huggingface.co/treadon/prompt-fungineer-355M) -- GPT-2 Medium (355M), trained in 2023 on web-scraped prompt data.
