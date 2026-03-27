"""
Generate 20,000 prompt pairs using Claude for Prompt Fungineer v2 training.
Outputs to HuggingFace dataset: treadon/prompt-fungineer-v2-training-data

Uses the Anthropic API to generate batches of prompt pairs.
Each pair: simple input (2-10 words) → detailed image prompt (50-200 words).

Usage:
    export ANTHROPIC_API_KEY=your_key
    python generate_dataset.py
"""

import os
import json
import time
import random
import anthropic
from pathlib import Path

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Category definitions with seed words to inspire variety
CATEGORIES = {
    "landscapes_nature": {
        "count": 2500,
        "seeds": [
            "mountain", "ocean", "forest", "desert", "waterfall", "lake", "canyon",
            "volcano", "glacier", "meadow", "river", "island", "cave", "cliff",
            "prairie", "tundra", "rainforest", "beach", "valley", "aurora",
            "sunrise", "storm", "fog", "snow", "autumn leaves", "wildflowers",
            "coral reef", "northern lights", "savanna", "rice terraces",
        ],
    },
    "portraits_people": {
        "count": 2500,
        "seeds": [
            "old woman", "child laughing", "dancer", "musician", "athlete",
            "chef cooking", "artist painting", "student reading", "couple walking",
            "fisherman", "monk meditating", "soldier", "nurse", "firefighter",
            "grandmother", "street performer", "yoga pose", "crying face",
            "celebration", "wedding", "graduation", "protest", "crowd",
            "silhouette", "reflection in mirror", "hands", "eyes close-up",
            "tattoo artist", "barber", "farmer",
        ],
    },
    "architecture_urban": {
        "count": 2000,
        "seeds": [
            "skyscraper", "old church", "bridge", "abandoned building", "neon signs",
            "alleyway", "subway station", "rooftop", "staircase", "window",
            "doorway", "market", "cafe", "library interior", "museum",
            "parking garage", "construction site", "lighthouse", "castle",
            "mosque", "temple", "pagoda", "gothic cathedral", "brutalist building",
            "art deco lobby", "train station", "greenhouse", "barn", "treehouse",
        ],
    },
    "animals_wildlife": {
        "count": 2000,
        "seeds": [
            "wolf", "eagle", "whale", "tiger", "butterfly", "owl", "dolphin",
            "elephant", "fox", "hummingbird", "jellyfish", "octopus", "lion",
            "penguin", "chameleon", "deer", "bear", "snake", "parrot",
            "seahorse", "gorilla", "flamingo", "dragonfly", "coral fish",
            "spider web", "cat sleeping", "dog running", "horse galloping",
        ],
    },
    "food_stilllife": {
        "count": 1500,
        "seeds": [
            "coffee cup", "sushi", "chocolate cake", "wine glass", "bread",
            "fruit bowl", "spices", "pizza", "ice cream", "tea ceremony",
            "cheese board", "fresh pasta", "herbs", "honey dripping",
            "street food", "breakfast spread", "cocktail", "farmers market",
            "kitchen counter", "old cookbook", "candles", "flowers in vase",
        ],
    },
    "abstract_artistic": {
        "count": 2000,
        "seeds": [
            "chaos", "serenity", "time passing", "dreams", "music visualized",
            "gravity", "infinity", "loneliness", "joy", "connection",
            "fractals", "geometric patterns", "paint splatter", "light trails",
            "smoke", "water droplets", "glass shards", "paper origami",
            "shadow play", "double exposure", "kaleidoscope", "prism light",
            "ink in water", "rust patterns", "ice crystals", "bubbles",
        ],
    },
    "scifi_fantasy": {
        "count": 2000,
        "seeds": [
            "spaceship", "alien planet", "robot", "cyberpunk city", "portal",
            "dragon", "wizard tower", "enchanted forest", "space station",
            "time machine", "underwater base", "floating islands", "crystal cave",
            "steampunk", "mech suit", "hologram", "warp drive", "terraforming",
            "ancient ruins on mars", "bioluminescent jungle", "ghost ship",
            "fairy village", "phoenix", "ice fortress", "void", "nebula",
        ],
    },
    "historical_vintage": {
        "count": 1500,
        "seeds": [
            "victorian street", "ancient rome", "samurai", "viking ship",
            "1920s jazz club", "wild west", "medieval market", "renaissance",
            "ancient egypt", "silk road", "industrial revolution", "prohibition era",
            "world war trenches", "ancient greek temple", "colonial town",
            "pirate ship", "gold rush", "old train", "vintage car", "typewriter",
        ],
    },
    "macro_closeup": {
        "count": 1500,
        "seeds": [
            "dewdrop", "insect eye", "flower petal", "circuit board", "fabric texture",
            "rust", "crystal", "feather", "leaf veins", "sand grains",
            "snowflake", "watch gears", "fingerprint", "soap bubble",
            "mushroom gills", "lichen", "seed pod", "spider silk", "moss",
            "wood grain", "marble texture", "oil on water", "frost pattern",
        ],
    },
    "vehicles_technology": {
        "count": 1000,
        "seeds": [
            "race car", "motorcycle", "sailboat", "helicopter", "bicycle",
            "submarine", "hot air balloon", "electric car", "rocket launch",
            "vintage airplane", "skateboard", "tractor", "drone", "train",
            "space shuttle", "hovercraft", "kayak", "tank", "ambulance",
        ],
    },
    "emotions_concepts": {
        "count": 1500,
        "seeds": [
            "hope", "fear", "love", "anger", "nostalgia", "wonder", "grief",
            "freedom", "imprisonment", "rebirth", "solitude", "celebration",
            "tension", "peace", "conflict", "discovery", "loss", "triumph",
            "vulnerability", "power", "patience", "urgency", "curiosity",
        ],
    },
}

SYSTEM_PROMPT = """You are an expert AI image prompt engineer. Your job is to take simple, short image descriptions and expand them into detailed, production-ready prompts that will produce stunning images when used with AI image generators (Midjourney, Stable Diffusion, DALL-E, Flux).

Rules:
- Output ONLY valid JSON. No other text.
- Each expanded prompt should be 50-200 words
- Include: scene details, lighting, composition, mood/atmosphere, and optionally camera/technical details
- Vary your style — don't start every prompt the same way
- Mix photographic and artistic styles
- Do NOT use real artist names or copyrighted character names
- Do NOT include NSFW content
- Be specific and evocative, not generic
- Include sensory details (textures, temperatures, sounds implied visually)"""

USER_PROMPT_TEMPLATE = """Generate {batch_size} prompt pairs for the category "{category}".

Use these seed words as starting inspiration (but create your own varied simple prompts, don't just use these directly):
{seeds}

Return a JSON array of objects, each with:
- "input": a simple 2-10 word image description
- "output": a detailed 50-200 word expanded prompt
- "style": one of "photographic", "cinematic", "painterly", "digital_art", "illustration"

Example format:
[
  {{
    "input": "lonely lighthouse",
    "output": "A solitary lighthouse stands against a turbulent twilight sky, its beam cutting through thick marine fog. The structure, weathered white paint peeling to reveal grey stone beneath, perches on volcanic basalt rocks battered by churning steel-grey waves. Cold blue-violet light from the disappearing sun contrasts with the warm amber glow from the lighthouse windows. Sea spray catches the last light, creating ephemeral diamonds in the air. A narrow stone path winds down to a small wooden dock where a forgotten rowboat rocks gently. The composition draws the eye upward along the lighthouse to the dramatic cloudscape above. Shot with natural dramatic lighting, wide-angle perspective, moody atmosphere.",
    "style": "cinematic"
  }}
]

Generate exactly {batch_size} pairs. Ensure variety in subject matter, style, and prompt structure."""


def generate_batch(client, category, seeds, batch_size=50):
    """Generate a batch of prompt pairs using Claude."""
    seed_sample = random.sample(seeds, min(len(seeds), 15))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                batch_size=batch_size,
                category=category.replace("_", " "),
                seeds=", ".join(seed_sample),
            ),
        }],
    )

    text = response.content[0].text.strip()

    # Extract JSON from response
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        pairs = json.loads(text)
        # Validate
        valid = []
        for p in pairs:
            if isinstance(p, dict) and "input" in p and "output" in p:
                if len(p["input"].split()) <= 15 and len(p["output"].split()) >= 20:
                    valid.append({
                        "input": p["input"].strip(),
                        "output": p["output"].strip(),
                        "category": category,
                        "style": p.get("style", "photographic"),
                    })
        return valid
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return []


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        return

    client = anthropic.Anthropic(api_key=api_key)

    all_pairs = []
    batch_size = 50  # pairs per API call

    for category, config in CATEGORIES.items():
        target = config["count"]
        seeds = config["seeds"]
        generated = 0
        attempts = 0

        print(f"\n{'='*60}")
        print(f"Category: {category} (target: {target})")
        print(f"{'='*60}")

        while generated < target and attempts < target // batch_size + 10:
            remaining = target - generated
            this_batch = min(batch_size, remaining)

            attempts += 1
            print(f"  Batch {attempts}: generating {this_batch} pairs...", end=" ", flush=True)

            try:
                pairs = generate_batch(client, category, seeds, this_batch)
                all_pairs.extend(pairs)
                generated += len(pairs)
                print(f"got {len(pairs)} (total: {generated}/{target})")

                # Save checkpoint every 500 pairs
                if len(all_pairs) % 500 < batch_size:
                    checkpoint_path = OUTPUT_DIR / "checkpoint.jsonl"
                    with open(checkpoint_path, "w") as f:
                        for p in all_pairs:
                            f.write(json.dumps(p) + "\n")
                    print(f"  Checkpoint saved: {len(all_pairs)} total pairs")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"error: {e}")
                time.sleep(5)

    # Final save
    output_path = OUTPUT_DIR / "training_data.jsonl"
    with open(output_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(all_pairs)} pairs")
    print(f"Saved to {output_path}")
    print(f"{'='*60}")

    # Stats
    by_category = {}
    by_style = {}
    for p in all_pairs:
        cat = p["category"]
        style = p.get("style", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
        by_style[style] = by_style.get(style, 0) + 1

    print("\nBy category:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nBy style:")
    for style, count in sorted(by_style.items(), key=lambda x: -x[1]):
        print(f"  {style}: {count}")

    # Upload to HuggingFace
    print("\nUploading to HuggingFace...")
    try:
        from datasets import Dataset
        ds = Dataset.from_list(all_pairs)
        ds.push_to_hub("treadon/prompt-fungineer-v2-training-data", private=False)
        print("Uploaded to treadon/prompt-fungineer-v2-training-data")
    except Exception as e:
        print(f"HF upload failed: {e}")
        print("Data is saved locally — upload manually later.")


if __name__ == "__main__":
    main()
