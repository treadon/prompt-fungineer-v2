# Prompt Fungineer v2: Distilling Prompt Engineering Into a Local Model

> Can you distill a frontier AI's prompt engineering ability into a small model that runs on a laptop?

## The Story So Far

In April 2023, I fine-tuned GPT-2 Medium (355M parameters) to expand simple image descriptions into detailed prompts for AI image generators. It worked — but it was limited. Generic expansions, occasional hallucinations, narrow vocabulary.

Three years later, the game has changed. Open-source 4B models now outperform 2023's 175B models. Distillation from frontier models is a proven technique. And image generators themselves have evolved — they respond to different vocabulary and techniques.

**This experiment: distill Claude's prompt engineering skill into Qwen3.5-4B via full fine-tuning on 10,000 Claude-generated prompt pairs.**

## Process Log

### Phase 1: Dataset Generation

**Method:** Claude (Sonnet/Opus) generates prompt pairs directly. No API — generated within Claude Code conversation.

**Target:** 10,000 pairs across 11 categories

**Categories:**
| Category | Target | Description |
|----------|--------|-------------|
| landscapes_nature | ~1,250 | Mountains, oceans, weather, seasons, geological |
| portraits_people | ~1,250 | Occupations, cultural, candid, group scenes |
| scifi_fantasy | ~1,000 | Alien worlds, magic, dystopia, space |
| abstract_artistic | ~1,000 | Color studies, surrealism, minimalism, texture |
| architecture_urban | ~900 | Interiors, ruins, modern, night scenes |
| animals_wildlife | ~800 | Underwater, insects, birds, baby animals |
| historical_vintage | ~700 | Specific eras, ancient civilizations, retro |
| food_stilllife | ~700 | Cooking action, markets, drinks, table settings |
| macro_closeup | ~700 | Natural patterns, mechanical, water, botanical |
| emotions_concepts | ~700 | Abstract emotions visualized as scenes |
| vehicles_technology | ~500 | Cars, aircraft, gadgets, machines |

**Style distribution across batches:**
- Photographic (with camera/lens specs)
- Cinematic (film-like composition and grading)
- Painterly (oil, watercolor, mixed media)
- Digital art (concept art, 3D render, matte painting)
- Illustration (editorial, graphic novel, poster)

**Quality controls:**
- No copyrighted names or characters
- No NSFW content
- Varied sentence structures (not all "A breathtaking image of...")
- Range of prompt lengths (50-150 words)
- Each pair validated: input ≤15 words, output ≥20 words

**Batch generation log:**

| Batch | Pairs | Focus | Status |
|-------|-------|-------|--------|
| 01 | 200 | landscapes, portraits, architecture, animals, food | generating... |
| 02 | 200 | scifi, abstract, historical, macro, emotions | generating... |
| 03 | 200 | landscapes, portraits, vehicles, animals, architecture | generating... |
| 04 | 200 | scifi, food, abstract, historical, macro | generating... |
| 05 | 400 | all categories evenly | generating... |
| 06 | 400 | unusual/creative inputs, global cultures, action | generating... |
| 07 | 400 | photographic realism, camera specs, film stocks | generating... |
| 08 | 400 | artistic/painterly styles, art movements | generating... |
| 09-N | TBD | remaining to reach 10K | pending |

**Final results: 9,400 valid pairs generated across 22 batches. Zero invalid entries.**

Dataset pushed to HuggingFace: [treadon/prompt-fungineer-v2-training-data](https://huggingface.co/datasets/treadon/prompt-fungineer-v2-training-data)

**Category distribution:**
| Category | Count |
|----------|-------|
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

**Style distribution:**
| Style | Count |
|-------|-------|
| photographic | 4,897 (52%) |
| cinematic | 1,362 (14%) |
| painterly | 1,336 (14%) |
| digital_art | 972 (10%) |
| illustration | 833 (9%) |

**Input length:** 1-9 words, avg 4.8
**Output length:** 53-181 words, avg 90.5

**Thematic coverage across batches:**
- Batches 1-4: Core categories (landscapes, portraits, architecture, animals, food, scifi, abstract, historical, macro)
- Batch 5: All categories evenly distributed
- Batch 6: Unusual/creative inputs, global cultures, action/motion
- Batch 7: Photographic realism with camera specs and film stocks
- Batch 8: Artistic/painterly styles and art movements
- Batch 9: World cultures and diversity (South Asian, East Asian, African, Middle Eastern, Latin American, Nordic, Mediterranean)
- Batch 10: Edge cases (single-word inputs, contradictions, abstract concepts, minimalist)
- Batch 11: Commercial/professional use cases (product, fashion, real estate, book covers, album art)
- Batch 12: Cinematic/narrative scenes (film noir, epic fantasy, horror, romance, documentary)
- Batch 13: Seasons and time of day (golden hour, blue hour, all four seasons, weather)
- Batch 14: Textures and materials (metal, natural, fabric, liquid, organic, manufactured)
- Batch 15: Color and light (monochrome, neon, pastel, saturated, dramatic contrast, iridescent)
- Batch 16: Scale and perspective (tilt-shift, aerial, macro, fisheye, split perspective)
- Batch 17: Weather and atmosphere (rain, snow, wind, fog, lightning, heat, clouds)
- Batch 18: Human activities and crafts (sports, craftsmanship, music, dance, science)
- Batch 19: Gap filling (illustration/painterly styles, short inputs, question-style inputs)
- Batch 20: Juxtaposition and contrast (old vs new, nature vs urban, warm vs cold, real vs surreal)
- Batch 21: Final gap filling (underwater, space, holidays, architectural details, sports action)
- Batch 22: Underrepresented areas (food, vehicles, whimsical inputs, mundane-made-beautiful)

### Phase 2: Training

**Status: IN PROGRESS**
**W&B Dashboard:** https://wandb.ai/actual-ritesh-org/prompt-fungineer-v2

**Final config (after several false starts — see issues below):**
- Base model: Qwen/Qwen3-0.6B (600M params)
- Method: Full fine-tune (all parameters unfrozen)
- Hardware: Apple M4 Pro, 64GB unified memory, MPS backend
- Framework: HuggingFace Trainer (not raw PyTorch — see issues)
- Epochs: 4
- Learning rate: 2e-5, cosine schedule with warmup (6%)
- Batch size: 1 per device, gradient accumulation 32 (effective batch 32)
- Precision: float32 (not BF16 — see issues)
- Gradient checkpointing: enabled (reduces activation memory)
- Max sequence length: 384 tokens
- Total optimization steps: 1,120
- Step speed: ~9.4s/step on MPS
- Expected time: ~2h 54m
- Peak memory: ~9.2 GB

**Why Qwen3-0.6B instead of 4B:**
The interesting experiment is NOT "bigger model = better." That's obvious. The question is: can a model the SAME SIZE as the original (355M vs 600M) produce dramatically better results thanks to 3 years of architectural improvements + better training data? That's science.

**Issues encountered (important lessons for Apple Silicon training):**

1. **Raw PyTorch training loop on MPS hung silently.** Wrote a manual training loop with `model.to("mps")` — the process started but never produced output. CPU usage was low (7-14%), RSS stayed at ~260MB (model weights never loaded to GPU). The first forward pass appeared to hang indefinitely, likely due to Metal shader compilation on Qwen3's architecture.
   - **Fix:** Switched to HuggingFace `Trainer`, which handles MPS device placement, shader compilation warmup, and memory management internally.

2. **BF16 training on MPS doesn't work.** Initial plan was BF16 to save memory (1.2GB model vs 2.4GB in float32). MPS supports BF16 for inference but training with BF16 gradients causes silent hangs or incorrect results.
   - **Fix:** Use float32 for training. Doubles memory usage but actually works.

3. **OOM at batch_size=4 with float32.** Memory budget was wrong — estimated 7-8GB but actual was 81GB at batch_size=4:
   - Model float32: 2.4 GB
   - Gradients: 2.4 GB
   - Optimizer (AdamW, 2 states): 4.8 GB
   - Activations at batch=4, seq=512: **~70 GB** ← this was the killer
   The activation memory scales with batch_size × seq_length × num_layers × hidden_dim. At float32, each activation tensor is 4 bytes. Qwen3-0.6B has 28 layers — the intermediate activations for 4 sequences of 512 tokens are enormous.
   - **Fix:** batch_size=1 + gradient_checkpointing=True + max_seq_len=384. Gradient checkpointing recomputes activations during backward pass instead of storing them, trading ~30% more compute time for ~70% less memory.

4. **HuggingFace DataCollatorForLanguageModeling can't handle custom labels.** We mask the user portion of the chat template so the model only trains on the assistant's output. The default collator doesn't support pre-computed labels with -100 masking.
   - **Fix:** Custom collate function that pads input_ids, attention_mask, and labels separately.

5. **`torch_dtype` deprecated in transformers.** Minor — renamed to `dtype` in newer versions.

**Key takeaway for Apple Silicon training:**
MPS is viable for training models up to ~1B parameters, but requires:
- HuggingFace Trainer (not raw PyTorch loops)
- float32 precision (not BF16)
- batch_size=1 with gradient accumulation
- Gradient checkpointing for anything over ~300M params
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to disable memory limits

The original GPT-2 355M trained on a 16GB NVIDIA card because (a) it used FP16 (NVIDIA supports this natively), (b) GPT-2 has fewer layers (24 vs 28), and (c) CUDA's memory management is much more mature than MPS. Apple Silicon has the raw memory (64GB) but MPS tooling is still catching up.

6. **Disk full during checkpoint save.** Each Qwen3-0.6B checkpoint at float32 is ~2.4GB. With `save_total_limit=3` plus the best model, checkpoints consumed 15GB. The drive filled up (926GB drive, only 347MB remaining — eaten by downloaded GGUF models from earlier experiments).
   - **Fix:** Cleaned up old model downloads (freed 53GB), reduced `save_total_limit=2`, increased `save_steps=500`.
   - **Lesson:** Monitor disk space before long training runs, especially when the same drive holds downloaded model weights from other experiments.

**Training run (final successful):**
- W&B: https://wandb.ai/actual-ritesh-org/prompt-fungineer-v2/runs/8o08hfkn
- Step speed: ~9.2s/step
- ETA: ~2h 50m for 1,120 steps (4 epochs)
- Peak memory: ~9.2GB RSS

### Phase 3: Evaluation

**Status: COMPLETE**

Ran 20 test prompts through both v1 (GPT-2 355M) and v2 (Qwen3-0.6B fine-tuned).

**Quantitative results:**

| Metric | v1 (GPT-2 355M) | v2 (Qwen3-0.6B) |
|--------|-----------------|------------------|
| Parameters | 354,823,168 | 596,049,920 |
| Avg words per output | 92 | 117 (+27%) |
| Avg generation time | 2.56s | 4.82s |
| Architecture year | 2019 | 2025 |
| Training data | Web-scraped | Claude-generated |
| Final train loss | unknown | 1.51 |
| Final eval loss | unknown | 2.67 |

**Qualitative findings:**

v2 improvements over v1:
- **More specific details:** v2 names actual locations ("Pacific Ocean", "Shibuya"), camera models ("Fuji X-T5"), and precise lighting conditions instead of generic adjectives
- **Better scene understanding:** v1 confused cherry blossoms with tropical rainforest, hallucinated "deer" instead of wolf. v2 gets the scene right
- **Technical photography vocabulary:** v2 includes focal lengths, aperture values, film stocks naturally — v1 mostly uses generic "wide-angle lens" or "telephoto lens"
- **More varied sentence structure:** v1 starts almost every output with "A breathtaking/captivating/whimsical image of..." v2 varies structure

v2 weaknesses:
- **Slower:** 4.82s vs 2.56s (1.9x slower) — larger model + more output tokens
- **Occasional hallucinations:** "chess grandmaster sits at a mahjong table" — mixes up related concepts
- **Chat template artifacts:** Sometimes includes "assistant\n" prefix in output (parsing issue, not model issue)
- **Repetitive within outputs:** v2 sometimes repeats phrases ("coral-encrusted columns and coral-encrusted...")

**Example comparisons (5 of 20):**

**"sunset over the ocean"**
- v1: "A breathtaking image of a tranquil, calm image of the horizon at sunset, revealing a serene, crystal-clear ocean with a tranquil white sand beach below."
- v2: "The Pacific Ocean at sunset, the sea surface in perfect mirror — the sky above reflected in the dark water below."
- **Winner: v2** — specific (Pacific Ocean), poetic (mirror surface), less generic

**"neon-lit street at night"**
- v1: "A captivating image of a bustling, neon-lit street at night, with pedestrians weaving through the mix of pedestrian traffic."
- v2: "A city street at 2am, every surface lit by a different coloured neon sign — red traffic lights, blue bus stop signs, green bus number boards."
- **Winner: v2** — specific time (2am), specific colors per sign, cinematic detail

**"cherry blossoms in rain"**
- v1: "A whimsical image of cherry blossom blossoms in a lush, tropical rainforest" ← WRONG (not a rainforest)
- v2: "A cherry blossom tree in full bloom photographed at the moment of rain, the petals floating and falling in the downpour. Fuji X-T5, 24mm f/4"
- **Winner: v2** — correct scene, camera specs, captures the moment

**"wolf howling at the moon"**
- v1: "A haunting image of a deer stalking the night sky" ← HALLUCINATED (deer, not wolf)
- v2: "A grey wolf stands in the moonlight, its amber eyes locked on the full moon. The moon is a perfect silver disc above the horizon."
- **Winner: v2** — correct animal, specific details (grey, amber eyes, silver disc)

**"rainy tokyo street"**
- v1: "A vibrant and dramatic image of a bustling tokyo street, showcasing the vibrant colors, textures"
- v2: "A narrow alley in Shibuya during the rain, the grey asphalt reflecting the distorted white shapes of the street lamps above."
- **Winner: v2** — specific district (Shibuya), specific reflections, atmospheric

### Phase 4: Publishing

**Status: IN PROGRESS**

- GitHub repo: treadon/prompt-fungineer-v2 (private) ✅
- HuggingFace model: treadon/prompt-fungineer-v2 (private) ✅
- HuggingFace dataset: treadon/prompt-fungineer-v2-training-data (public) ✅
- Model card: in progress
- Blog post: in progress
- HuggingFace Space: pending
- Tweet thread: pending

## Key Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Student model | Qwen3.5-4B | Best instruction following at this size, same family as trending HF models |
| Teacher | Claude (Sonnet + Opus) | Highest quality prompt expansion, we can control format exactly |
| Training method | Full fine-tune | Specialist model, not generalist — want full commitment to the task |
| Dataset size | 10K pairs | Enough for full fine-tune without severe overfitting at 2-3 epochs |
| Dataset source | Generated by Claude in-conversation | No API costs, highest quality, format control |

## What We'll Learn

1. **Scale jump:** Does going from 355M → 4B (11x) meaningfully improve prompt quality?
2. **Teacher quality:** Does Claude-generated training data produce better students than web-scraped data?
3. **Full fine-tune vs LoRA:** For single-task models, is full fine-tuning worth the extra compute?
4. **Prompt evolution:** Do 2026 prompts differ from 2023 prompts in structure and vocabulary?
5. **Distillation ceiling:** How close can a 4B student get to its Claude teacher?

## Conclusions

### 1. Architecture generation matters more than raw scale

Qwen3-0.6B (2025) with 596M parameters dramatically outperforms GPT-2 (2019) at 355M — not because it's 1.7x bigger, but because of 6 years of architectural improvements: GQA attention, RoPE positional encoding, 151K vocabulary (vs 50K), and multilingual pretraining.

### 2. Training data quality matters more than quantity

The original fungineer trained on thousands of web-scraped prompts. v2 trains on only 9,400 Claude-generated pairs — yet produces dramatically more specific, varied, and technically accurate outputs. Clean data from a frontier model beats noisy web scraping.

### 3. The distillation ceiling is real but high

v2 captures Claude's vocabulary, structure, and technical specificity. But it still hallucinates (chess → mahjong) and sometimes repeats itself. A 600M model can learn the *pattern* of expert prompt engineering but not the *reasoning*.

### 4. MPS training is viable but painful

Full fine-tuning on Apple Silicon works with workarounds: float32 (not BF16), batch_size=1 + gradient accumulation, gradient checkpointing, HF Trainer (not raw PyTorch). The ~3 hour training time is acceptable. CUDA would do it in 20 minutes.

### 5. "Same size, better everything else" is the right experiment

Going from 355M to 4B would have shown obvious improvement but taught us nothing. Staying at the same size class and changing only architecture + training data isolates what actually matters.

### 6. v2 wins on specificity, v1 wins on speed

v2: 27% more words, names specific locations/cameras/lighting, correct scene understanding.
v1: 1.9x faster, no chat template artifacts, simpler to deploy.

For production use, v2 is clearly better. For understanding what drives quality, the answer is: architecture + data quality > parameter count.
