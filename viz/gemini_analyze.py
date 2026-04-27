"""Send prediction vs GT visualizations to Gemini for visual pattern analysis.

Renders grid comparisons as images and asks Gemini to identify spatial
patterns in prediction errors that might be hard to spot in numbers.

Usage:
    cd tasks/astar-island
    GOOGLE_API_KEY=... python viz/gemini_analyze.py [round_num] [seed]
    GOOGLE_API_KEY=... python viz/gemini_analyze.py 7 0       # analyze R7 seed 0
    GOOGLE_API_KEY=... python viz/gemini_analyze.py all        # analyze worst seed per round
"""

import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import build_prediction, load_calibration, compute_obs_stats
from utils import normalize_prediction, load_observations
from evaluate import compute_score, kl_divergence, entropy

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

ROUNDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
}

# Color scheme matching viz.html
COLORS = {
    0: (168, 216, 234),  # Empty - light blue
    1: (255, 153, 51),   # Settlement - orange
    2: (204, 51, 51),    # Port - red
    3: (136, 136, 136),  # Ruin - gray
    4: (51, 170, 51),    # Forest - green
    5: (102, 85, 51),    # Mountain - brown
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def render_grid(grid_data, cell_size=12, title=""):
    """Render a 2D grid as a PIL image. grid_data can be:
    - initial grid (int codes)
    - argmax of probability tensor
    - KL divergence heatmap (float)
    """
    h, w = len(grid_data), len(grid_data[0])
    img_w = w * cell_size + 2
    img_h = h * cell_size + 20  # extra space for title
    img = Image.new("RGB", (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((4, 2), title, fill=(200, 200, 200))

    for y in range(h):
        for x in range(w):
            val = grid_data[y][x]
            if isinstance(val, (int, np.integer)):
                color = COLORS.get(int(val), (50, 50, 50))
            else:
                # KL heatmap: 0=white, 2+=red
                intensity = min(1.0, float(val) / 2.0)
                r = int(255 * intensity)
                g = int(255 * (1 - intensity))
                b = int(255 * (1 - intensity))
                color = (r, g, b)

            x0 = x * cell_size + 1
            y0 = y * cell_size + 18
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=color)

    return img


def initial_to_class(code):
    """Map initial terrain code to class index."""
    mapping = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    return mapping.get(code, 0)


def build_comparison_image(round_num, seed, pred, gt, initial_grid, score):
    """Build a 4-panel comparison image."""
    h, w = len(initial_grid), len(initial_grid[0])

    # Panel 1: Initial terrain
    init_classes = [[initial_to_class(initial_grid[y][x]) for x in range(w)] for y in range(h)]
    img_init = render_grid(init_classes, title=f"Initial (R{round_num} s{seed})")

    # Panel 2: Prediction argmax
    pred_argmax = np.argmax(pred, axis=-1).tolist()
    img_pred = render_grid(pred_argmax, title=f"Prediction (score={score:.1f})")

    # Panel 3: GT argmax
    gt_argmax = np.argmax(gt, axis=-1).tolist()
    img_gt = render_grid(gt_argmax, title="Ground Truth")

    # Panel 4: KL error heatmap
    kl_map = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            if initial_grid[y][x] not in (10, 5):
                kl_map[y, x] = kl_divergence(gt[y, x], pred[y, x])
    img_kl = render_grid(kl_map.tolist(), title="KL Error (red=bad)")

    # Combine 4 panels into one image (2x2)
    panel_w = img_init.width
    panel_h = img_init.height
    combined = Image.new("RGB", (panel_w * 2 + 4, panel_h * 2 + 4), (20, 20, 20))
    combined.paste(img_init, (0, 0))
    combined.paste(img_pred, (panel_w + 4, 0))
    combined.paste(img_gt, (0, panel_h + 4))
    combined.paste(img_kl, (panel_w + 4, panel_h + 4))

    return combined


def analyze_with_gemini(image, round_num, seed, score, per_class_errors):
    """Send image to Gemini for visual analysis."""
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY env var")
        return None

    client = genai.Client(api_key=api_key)

    error_summary = "\n".join(
        f"  {CLASS_NAMES[i]}: avg_KL={err:.4f}"
        for i, err in enumerate(per_class_errors)
    )

    prompt = f"""You are analyzing predictions from a Norse civilization simulator competition.

The image shows 4 panels for Round {round_num}, Seed {seed} (score: {score:.1f}/100):
- Top-left: INITIAL terrain map (what the map starts as)
- Top-right: MODEL PREDICTION (what our model predicts after 50 years of simulation)
- Bottom-left: GROUND TRUTH (actual outcome from hundreds of simulations)
- Bottom-right: KL DIVERGENCE ERROR (red = high error, white = good prediction)

Color coding:
- Light blue = Empty/Ocean/Plains
- Orange = Settlement
- Red = Port (coastal settlement with harbor)
- Gray = Ruin (collapsed settlement)
- Green = Forest
- Brown = Mountain

Per-class error breakdown:
{error_summary}

Please analyze:
1. WHERE are the errors concentrated spatially? (e.g., near coastlines, in specific regions, near settlements)
2. WHAT types of terrain transitions is the model getting wrong? (e.g., predicting too many settlements, missing ports)
3. Are there PATTERNS in the error heatmap? (e.g., ring around settlements, coastal strip, isolated patches)
4. What specific improvements would reduce the largest errors?
5. Compare the prediction and GT — what does the model "understand" well vs poorly?

Be specific about spatial locations (top-left, center, along the coast, etc.) and terrain types."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt],
    )

    return response.text


def analyze_round(round_num, seed=0):
    """Build prediction, render comparison, send to Gemini."""
    print(f"\n{'='*60}")
    print(f"Analyzing Round {round_num}, Seed {seed}")
    print(f"{'='*60}")

    # Load data
    with open(os.path.join(DATA_DIR, f"round{round_num}_initial.json")) as f:
        info = json.load(f)

    gt = np.load(os.path.join(DATA_DIR, f"gt_r{round_num}_seed{seed}.npy"))
    grid = info["initial_states"][seed]["grid"]
    settlements = info["initial_states"][seed]["settlements"]

    # Load observations
    round_id = ROUNDS.get(round_num, "")
    all_obs = load_observations(round_id) if round_id else []
    same_seed_obs = [o for o in all_obs if o.get("seed_index") == seed]

    all_grids = [info["initial_states"][i]["grid"] for i in range(5)]
    cal = load_calibration()

    # Build prediction
    pred = build_prediction(
        initial_grid=grid, settlements=settlements,
        observations=same_seed_obs, seed_index=seed,
        all_initial_grids=all_grids, all_observations=all_obs,
        calibration=cal,
    )
    pred = normalize_prediction(pred)

    score = compute_score(pred, gt)
    print(f"Score: {score:.2f}")

    # Per-class errors
    h, w = len(grid), len(grid[0])
    cls_kl = np.zeros(6)
    cls_count = np.zeros(6)
    for y in range(h):
        for x in range(w):
            if grid[y][x] not in (10, 5):
                cell_kl = kl_divergence(gt[y, x], pred[y, x])
                cell_ent = entropy(gt[y, x])
                if cell_ent > 0:
                    for c in range(6):
                        if gt[y, x, c] > 0:
                            cls_kl[c] += gt[y, x, c] * np.log(gt[y, x, c] / max(pred[y, x, c], 1e-10))
                            cls_count[c] += 1

    per_class_errors = cls_kl / np.maximum(cls_count, 1)

    # Render comparison image
    img = build_comparison_image(round_num, seed, pred, gt, grid, score)
    img_path = os.path.join(os.path.dirname(__file__), f"analysis_r{round_num}_s{seed}.png")
    img.save(img_path)
    print(f"Saved: {img_path}")

    # Send to Gemini
    print("Sending to Gemini for analysis...")
    analysis = analyze_with_gemini(img, round_num, seed, score, per_class_errors)

    if analysis:
        print(f"\n--- Gemini Analysis ---\n{analysis}")

        # Save analysis
        txt_path = os.path.join(os.path.dirname(__file__), f"analysis_r{round_num}_s{seed}.txt")
        with open(txt_path, "w") as f:
            f.write(analysis)
        print(f"Saved: {txt_path}")

    return score, analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: GOOGLE_API_KEY=... python viz/gemini_analyze.py <round|all> [seed]")
        print("  python viz/gemini_analyze.py 7 0     # R7 seed 0")
        print("  python viz/gemini_analyze.py all      # worst seed per round")
        return

    if sys.argv[1] == "all":
        for rnum in sorted(ROUNDS.keys()):
            try:
                analyze_round(rnum, seed=0)
            except Exception as e:
                print(f"R{rnum}: error - {e}")
    else:
        rnum = int(sys.argv[1])
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        analyze_round(rnum, seed)


if __name__ == "__main__":
    main()
