"""
Prompt Fungineer v2 Demo
Compare v1 (GPT-2 355M) vs v2 (Qwen3-0.6B) prompt expansion side-by-side.
"""

import re
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
V1_REPO = "treadon/prompt-fungineer-355M"
V2_REPO = "treadon/prompt-fungineer-v2"

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
print("Loading v1 (GPT-2 355M)...", flush=True)
v1_tokenizer = AutoTokenizer.from_pretrained("gpt2")
v1_model = AutoModelForCausalLM.from_pretrained(V1_REPO)
v1_model.eval()
v1_params = sum(p.numel() for p in v1_model.parameters())
print(f"  v1 params: {v1_params:,}", flush=True)

print("Loading v2 (Qwen3-0.6B)...", flush=True)
v2_tokenizer = AutoTokenizer.from_pretrained(V2_REPO)
v2_model = AutoModelForCausalLM.from_pretrained(V2_REPO, torch_dtype=torch.float32)
v2_model.eval()
v2_params = sum(p.numel() for p in v2_model.parameters())
print(f"  v2 params: {v2_params:,}", flush=True)

print("Models loaded!", flush=True)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def format_v1_output(raw: str) -> str:
    """Parse v1's structured BRF/POS/ENH output format."""
    try:
        pattern = r"(BRF:|POS:|ENH:|INS:|NEG:)\s*(.*?)(?=\s*(?:BRF:|POS:|ENH:|INS:|NEG:)|$)"
        matches = re.findall(pattern, raw)
        vals = {key: value.strip() for key, value in matches}
        result = vals.get("POS:", "")
        if "ENH:" in vals:
            result += " " + vals["ENH:"]
        return result.strip()
    except Exception:
        return raw


def generate_v1(prompt: str) -> str:
    """Generate an expanded prompt with v1 (GPT-2)."""
    input_text = f"BRF: {prompt} POS:"
    inputs = v1_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = v1_model.generate(
            **inputs,
            max_length=256,
            do_sample=True,
            top_k=100,
            top_p=0.95,
            temperature=0.85,
            pad_token_id=v1_tokenizer.eos_token_id,
        )
    raw = v1_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_v1_output(raw)


def generate_v2(prompt: str) -> str:
    """Generate an expanded prompt with v2 (Qwen3-0.6B)."""
    input_text = (
        f"<|im_start|>user\n"
        f"Expand this into a detailed image generation prompt: {prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = v2_tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = v2_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=v2_tokenizer.eos_token_id,
        )
    full = v2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant response
    if "<|im_start|>assistant" in full:
        response = full.split("<|im_start|>assistant")[-1].strip()
        if response.startswith("\n"):
            response = response[1:]
        for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
            response = response.split(marker)[0]
        return response.strip()
    return full.strip()


def fungineer(prompt: str):
    """Run both models and return their outputs."""
    if not prompt or not prompt.strip():
        return "", ""
    prompt = prompt.strip()
    v1_out = generate_v1(prompt)
    v2_out = generate_v2(prompt)
    return v1_out, v2_out


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["a cat on a mountain"],
    ["sunset over the ocean"],
    ["abandoned spaceship in a desert"],
    ["neon-lit street at night"],
    ["old man playing chess in the park"],
]

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="pink",
    ),
    title="Prompt Fungineer v2",
) as demo:
    gr.Markdown(
        """
        # Prompt Fungineer v2
        **Turn simple prompts into detailed image-generation prompts.**

        Type a short description and see how two models expand it side-by-side:
        - **v1** -- GPT-2 (355 M params), trained on structured BRF/POS/ENH data
        - **v2** -- Qwen3-0.6B (596 M params), distilled from Claude on chat-formatted pairs
        """
    )

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Your prompt",
            placeholder="e.g. a robot painting in a garden",
            lines=1,
            scale=4,
        )
        run_btn = gr.Button("Fungineer it!", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### v1 -- GPT-2 ({v1_params:,} params)")
            v1_output = gr.Textbox(
                label="v1 expansion",
                lines=8,
                interactive=False,
                show_copy_button=True,
            )
        with gr.Column():
            gr.Markdown(f"### v2 -- Qwen3-0.6B ({v2_params:,} params)")
            v2_output = gr.Textbox(
                label="v2 expansion",
                lines=8,
                interactive=False,
                show_copy_button=True,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=prompt_input,
        outputs=[v1_output, v2_output],
        fn=fungineer,
        cache_examples=False,
    )

    # Wire up events
    run_btn.click(fn=fungineer, inputs=prompt_input, outputs=[v1_output, v2_output])
    prompt_input.submit(fn=fungineer, inputs=prompt_input, outputs=[v1_output, v2_output])

    IMG_BASE = "https://raw.githubusercontent.com/treadon/prompt-fungineer-v2/main/images"

    gr.Markdown("---\n### Visual Proof: Better Prompts → Better Images\nSame image generator (Grok Imagine), different prompts. Left = basic input, middle = v1 prompt, right = v2 prompt.")

    for slug, label in [
        ("sunset_over_the_ocean", "sunset over the ocean"),
        ("cherry_blossoms_in_rain", "cherry blossoms in rain"),
        ("wolf_howling_at_the_moon", "wolf howling at the moon"),
        ("neon-lit_street_at_night", "neon-lit street at night"),
        ("rainy_tokyo_street", "rainy tokyo street"),
    ]:
        gr.Markdown(f"**\"{label}\"**")
        with gr.Row():
            gr.Image(f"{IMG_BASE}/{slug}_basic.jpg", label="Basic input", show_download_button=False)
            gr.Image(f"{IMG_BASE}/{slug}_v1.jpg", label="v1 (GPT-2)", show_download_button=False)
            gr.Image(f"{IMG_BASE}/{slug}_v2.jpg", label="v2 (Qwen3)", show_download_button=False)

    gr.Markdown(
        """
        ---
        *Models: [treadon/prompt-fungineer-355M](https://huggingface.co/treadon/prompt-fungineer-355M)
        and [treadon/prompt-fungineer-v2](https://huggingface.co/treadon/prompt-fungineer-v2) |
        [GitHub](https://github.com/treadon/prompt-fungineer-v2) |
        [Blog](https://riteshkhanna.com/blog/prompt-fungineer-v2)*
        """
    )

demo.launch()
