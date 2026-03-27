"""Regenerate v1 images with properly cleaned prompts (no BRF leaking)."""
import os, json, re, time, requests

def load_api_key():
    with open(os.path.expanduser("~/Dev/verytoronto/.env.local")) as f:
        for line in f:
            if line.startswith("KIE_AI_API_KEY="):
                return line.strip().split("=", 1)[1]

API_KEY = load_api_key()
BASE_URL = "https://api.kie.ai/api/v1/jobs"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def generate(prompt):
    resp = requests.post(f"{BASE_URL}/createTask", headers=HEADERS,
        json={"model": "grok-imagine/text-to-image", "input": {"prompt": prompt, "aspect_ratio": "1:1"}})
    data = resp.json()
    task_id = data.get("taskId") or (data.get("data", {}) or {}).get("taskId") or (data.get("data", {}) or {}).get("task_id")
    if not task_id:
        print(f"  No taskId: {data}", flush=True)
        return []
    for _ in range(30):
        time.sleep(3)
        resp = requests.get(f"{BASE_URL}/recordInfo", headers=HEADERS, params={"taskId": task_id})
        data = resp.json()
        status = (data.get("status") or (data.get("data", {}) or {}).get("state") or "").lower()
        if status in ("success", "completed"):
            rj = (data.get("data", {}) or {}).get("resultJson")
            if rj:
                return json.loads(rj).get("resultUrls", [])
            img = (data.get("output", {}) or {}).get("image_url")
            return [img] if img else []
        if status == "failed":
            return []
    return []

with open("evaluation_results.json") as f:
    eval_data = json.load(f)

prompts = ["sunset over the ocean", "cherry blossoms in rain", "wolf howling at the moon", "neon-lit street at night", "rainy tokyo street"]

for inp in prompts:
    match = next(r for r in eval_data["results"] if r["input"] == inp)
    v1 = match["v1_output"]

    # Properly extract POS + ENH only (no BRF prefix)
    pos = re.search(r'POS:\s*(.*?)(?=\s*(?:ENH:|INS:|NEG:)|$)', v1, re.DOTALL)
    enh = re.search(r'ENH:\s*(.*?)(?=\s*(?:INS:|NEG:)|$)', v1, re.DOTALL)
    clean = (pos.group(1).strip() if pos else "") + (" " + enh.group(1).strip() if enh else "")

    slug = inp.replace(" ", "_")
    print(f"\n{inp}", flush=True)
    print(f"  Prompt: {clean[:120]}...", flush=True)

    urls = generate(clean)
    if urls:
        resp = requests.get(urls[0], timeout=30)
        path = f"images/{slug}_v1.jpg"
        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved: {path} ({len(resp.content)//1024}KB)", flush=True)
    else:
        print(f"  FAILED", flush=True)
    time.sleep(2)

print("\nDone!", flush=True)
