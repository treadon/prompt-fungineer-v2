"""
Generate images for the 5 test prompts using Grok Imagine via kie.ai API.
For each prompt: generate with the basic input, v1 output, and v2 output.
Saves all 4 Grok outputs per request but uses the first one.
"""

import os
import json
import time
import requests
from pathlib import Path

# Load API key from verytoronto env
def load_api_key():
    env_path = os.path.expanduser("~/Dev/verytoronto/.env.local")
    with open(env_path) as f:
        for line in f:
            if line.startswith("KIE_AI_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise ValueError("KIE_AI_API_KEY not found")

API_KEY = load_api_key()
BASE_URL = "https://api.kie.ai/api/v1/jobs"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_task(prompt, aspect_ratio="1:1"):
    """Create an image generation task."""
    resp = requests.post(
        f"{BASE_URL}/createTask",
        headers=HEADERS,
        json={
            "model": "grok-imagine/text-to-image",
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
            },
        },
    )
    data = resp.json()
    # Handle multiple response formats
    task_id = (
        data.get("taskId")
        or (data.get("data", {}) or {}).get("taskId")
        or (data.get("data", {}) or {}).get("task_id")
    )
    if not task_id:
        print(f"  ERROR: No taskId in response: {data}", flush=True)
        return None
    return task_id


def poll_task(task_id, max_attempts=30, interval=3):
    """Poll until task completes."""
    for attempt in range(max_attempts):
        time.sleep(interval)
        resp = requests.get(
            f"{BASE_URL}/recordInfo",
            headers=HEADERS,
            params={"taskId": task_id},
        )
        data = resp.json()

        # Check status
        status = (
            data.get("status")
            or (data.get("data", {}) or {}).get("state")
            or ""
        ).lower()

        if status in ("success", "completed"):
            # Extract image URLs
            urls = []

            # Format 1: resultJson with resultUrls array
            result_json_str = (data.get("data", {}) or {}).get("resultJson")
            if result_json_str:
                try:
                    result_json = json.loads(result_json_str)
                    urls = result_json.get("resultUrls", [])
                except json.JSONDecodeError:
                    pass

            # Format 2: direct image_url
            if not urls:
                img_url = (data.get("output", {}) or {}).get("image_url")
                if img_url:
                    urls = [img_url]

            return urls

        if status == "failed":
            print(f"  Task failed: {data}", flush=True)
            return []

        if attempt % 5 == 4:
            print(f"    Polling... attempt {attempt + 1}/{max_attempts}", flush=True)

    print(f"  Timed out after {max_attempts} attempts", flush=True)
    return []


def download_image(url, path):
    """Download image to local file."""
    resp = requests.get(url, timeout=30)
    with open(path, "wb") as f:
        f.write(resp.content)
    return len(resp.content)


def generate_and_save(prompt, name, subfolder):
    """Generate image from prompt and save all results."""
    out_dir = OUTPUT_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating: {name}...", flush=True)
    print(f"    Prompt: {prompt[:100]}...", flush=True)

    task_id = create_task(prompt)
    if not task_id:
        return None

    print(f"    Task ID: {task_id}", flush=True)
    urls = poll_task(task_id)

    if not urls:
        print(f"    No images returned!", flush=True)
        return None

    print(f"    Got {len(urls)} images", flush=True)

    # Save all images
    for i, url in enumerate(urls):
        path = out_dir / f"{name}_{i+1}.jpg"
        size = download_image(url, path)
        print(f"    Saved: {path} ({size/1024:.0f}KB)", flush=True)

    # Return path of first image (the one we'll use)
    return str(out_dir / f"{name}_1.jpg")


def main():
    # Load evaluation results for the actual v1/v2 outputs
    with open("evaluation_results.json") as f:
        eval_data = json.load(f)

    # The 5 test prompts we're using
    test_inputs = [
        "sunset over the ocean",
        "cherry blossoms in rain",
        "wolf howling at the moon",
        "neon-lit street at night",
        "rainy tokyo street",
    ]

    results = []

    for input_prompt in test_inputs:
        print(f"\n{'='*60}", flush=True)
        print(f"Input: \"{input_prompt}\"", flush=True)
        print(f"{'='*60}", flush=True)

        # Find matching evaluation result
        match = next((r for r in eval_data["results"] if r["input"] == input_prompt), None)
        if not match:
            print(f"  Skipping - not found in evaluation results", flush=True)
            continue

        # Clean v2 output (remove chat template artifacts)
        v2_output = match["v2_output"]
        for marker in ["user\nExpand this into a detailed image generation prompt:", "assistant\n", "assistant"]:
            if v2_output.startswith(marker):
                v2_output = v2_output[len(marker):].strip()
        if input_prompt in v2_output:
            v2_output = v2_output.split(input_prompt, 1)[-1].strip()
            if v2_output.startswith("\nassistant"):
                v2_output = v2_output[len("\nassistant"):].strip()

        slug = input_prompt.replace(" ", "_")

        entry = {"input": input_prompt, "images": {}}

        # 1. Generate from basic input
        path = generate_and_save(input_prompt, f"{slug}_basic", slug)
        entry["images"]["basic"] = path
        time.sleep(2)

        # 2. Generate from v1 output
        v1_clean = match["v1_output"]
        # Remove BRF:/POS: tags
        for tag in ["BRF:", "POS:", "ENH:", "INS:", "NEG:"]:
            v1_clean = v1_clean.replace(tag, "")
        v1_clean = " ".join(v1_clean.split())  # normalize whitespace

        path = generate_and_save(v1_clean, f"{slug}_v1", slug)
        entry["images"]["v1"] = path
        time.sleep(2)

        # 3. Generate from v2 output
        path = generate_and_save(v2_output, f"{slug}_v2", slug)
        entry["images"]["v2"] = path
        time.sleep(2)

        results.append(entry)
        print(f"  Done: {input_prompt}", flush=True)

    # Save manifest
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"All done! {len(results)} prompts × 3 versions = {len(results)*3} image sets", flush=True)
    print(f"Images saved to {OUTPUT_DIR}/", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
