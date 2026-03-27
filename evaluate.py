"""
Evaluate Prompt Fungineer v2 against v1 (GPT-2 355M).
Generates prompt expansions for test inputs and compares quality.
"""

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

V1_MODEL = "treadon/prompt-fungineer-355M"
V2_MODEL = "checkpoints/prompt-fungineer-v2/best"

TEST_PROMPTS = [
    "a cat on a mountain",
    "sunset over the ocean",
    "abandoned spaceship",
    "old man playing chess",
    "cherry blossoms in rain",
    "underwater city",
    "wolf howling at the moon",
    "neon-lit street at night",
    "ancient temple in jungle",
    "astronaut on Mars",
    "lonely lighthouse",
    "jazz musician",
    "frozen waterfall",
    "child flying a kite",
    "steampunk clock tower",
    "rainy tokyo street",
    "desert oasis",
    "grandmother baking bread",
    "northern lights",
    "robot in a garden",
]

import re

def format_v1_output(raw):
    """Parse v1's structured output format."""
    try:
        pattern = r'(BRF:|POS:|ENH:|INS:|NEG:) (.*?)(?= (?:BRF:|POS:|ENH:|INS:|NEG:)|$)'
        matches = re.findall(pattern, raw)
        vals = {key: value.strip() for key, value, ex in matches}
        result = vals.get("POS:", "")
        if "ENH:" in vals:
            result += " " + vals["ENH:"]
        return result
    except:
        return raw


def generate_v1(prompt, model, tokenizer):
    """Generate with v1 (GPT-2 format)."""
    input_text = f"BRF: {prompt} POS:"
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=256, do_sample=True,
            top_k=100, top_p=0.95, temperature=0.85,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_v1_output(raw)


def generate_v2(prompt, model, tokenizer):
    """Generate with v2 (chat format)."""
    input_text = (
        f"<|im_start|>user\n"
        f"Expand this into a detailed image generation prompt: {prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, do_sample=True,
            top_k=50, top_p=0.9, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant response
    if "<|im_start|>assistant" in full:
        response = full.split("<|im_start|>assistant")[-1].strip()
        if response.startswith("\n"):
            response = response[1:]
        # Remove any trailing special tokens
        for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
            response = response.split(marker)[0]
        return response.strip()
    return full.strip()


def main():
    print("=" * 60, flush=True)
    print("PROMPT FUNGINEER EVALUATION", flush=True)
    print("v1 (GPT-2 355M) vs v2 (Qwen3-0.6B)", flush=True)
    print("=" * 60, flush=True)

    # Load v1
    print("\nLoading v1 (GPT-2 355M)...", flush=True)
    v1_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    v1_model = AutoModelForCausalLM.from_pretrained(V1_MODEL)
    v1_model.eval()
    print(f"  Params: {sum(p.numel() for p in v1_model.parameters()):,}", flush=True)

    # Load v2
    print("Loading v2 (Qwen3-0.6B fine-tuned)...", flush=True)
    v2_tokenizer = AutoTokenizer.from_pretrained(V2_MODEL)
    v2_model = AutoModelForCausalLM.from_pretrained(V2_MODEL, torch_dtype=torch.float32)
    v2_model.eval()
    print(f"  Params: {sum(p.numel() for p in v2_model.parameters()):,}", flush=True)

    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─' * 60}", flush=True)
        print(f"[{i+1}/{len(TEST_PROMPTS)}] Input: \"{prompt}\"", flush=True)
        print(f"{'─' * 60}", flush=True)

        # v1
        t0 = time.time()
        v1_output = generate_v1(prompt, v1_model, v1_tokenizer)
        v1_time = time.time() - t0

        # v2
        t0 = time.time()
        v2_output = generate_v2(prompt, v2_model, v2_tokenizer)
        v2_time = time.time() - t0

        print(f"\nv1 ({v1_time:.1f}s): {v1_output[:200]}...", flush=True)
        print(f"\nv2 ({v2_time:.1f}s): {v2_output[:200]}...", flush=True)

        results.append({
            "input": prompt,
            "v1_output": v1_output,
            "v2_output": v2_output,
            "v1_time": v1_time,
            "v2_time": v2_time,
            "v1_words": len(v1_output.split()),
            "v2_words": len(v2_output.split()),
        })

    # Summary stats
    print(f"\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    v1_avg_time = sum(r["v1_time"] for r in results) / len(results)
    v2_avg_time = sum(r["v2_time"] for r in results) / len(results)
    v1_avg_words = sum(r["v1_words"] for r in results) / len(results)
    v2_avg_words = sum(r["v2_words"] for r in results) / len(results)

    print(f"v1 avg time: {v1_avg_time:.2f}s, avg words: {v1_avg_words:.0f}", flush=True)
    print(f"v2 avg time: {v2_avg_time:.2f}s, avg words: {v2_avg_words:.0f}", flush=True)

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "v1_model": V1_MODEL,
            "v2_model": V2_MODEL,
            "v1_params": sum(p.numel() for p in v1_model.parameters()),
            "v2_params": sum(p.numel() for p in v2_model.parameters()),
            "v1_avg_time": v1_avg_time,
            "v2_avg_time": v2_avg_time,
            "v1_avg_words": v1_avg_words,
            "v2_avg_words": v2_avg_words,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to evaluation_results.json", flush=True)


if __name__ == "__main__":
    main()
