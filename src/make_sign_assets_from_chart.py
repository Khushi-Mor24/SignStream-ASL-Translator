import os
from pathlib import Path
from PIL import Image

# === CONFIG ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHART_PATH = PROJECT_ROOT / "asl_chart.png"   # change to .jpg if needed
OUT_DIR = PROJECT_ROOT / "assets" / "signs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# This chart layout (your image) is:
# row1: a b c d e f        -> 6 letters
# row2: g h i j k          -> 5 letters
# row3: l m n o p          -> 5 letters
# row4: q r s t u          -> 5 letters
# row5: v w x y z          -> 5 letters

rows_letters = [
    list("abcdef"),
    list("ghijk"),
    list("lmnop"),
    list("qrstu"),
    list("vwxyz"),
]

def main():
    if not CHART_PATH.exists():
        print(f"Chart image not found at: {CHART_PATH}")
        return

    img = Image.open(CHART_PATH).convert("RGB")
    W, H = img.size
    print(f"Loaded chart: {CHART_PATH.name}, size = {W}x{H}")

    # we leave a little margin around each cell so we don't cut too tightly
    top_margin_frac = 0.03
    bottom_margin_frac = 0.15   # bottom area has the text "AMERICAN SIGN LANGUAGE"
    left_margin_frac = 0.03
    right_margin_frac = 0.03

    usable_top = int(H * top_margin_frac)
    usable_bottom = int(H * (1 - bottom_margin_frac))
    usable_height = usable_bottom - usable_top

    # 5 rows of letters
    n_rows = 5
    row_h = usable_height / n_rows

    for r, letters in enumerate(rows_letters):
        # y-coordinates for this row
        y1 = int(usable_top + r * row_h)
        y2 = int(usable_top + (r + 1) * row_h)

        n_cols = len(letters)
        col_w = (W * (1 - left_margin_frac - right_margin_frac)) / n_cols
        x_start = int(W * left_margin_frac)

        for c, letter in enumerate(letters):
            x1 = int(x_start + c * col_w)
            x2 = int(x_start + (c + 1) * col_w)

            # slightly shrink to avoid borders touching neighbors
            pad_x = int(col_w * 0.08)
            pad_y = int(row_h * 0.12)

            x1p = max(0, x1 + pad_x)
            x2p = min(W, x2 - pad_x)
            y1p = max(0, y1 + pad_y)
            y2p = min(H, y2 - pad_y)

            crop = img.crop((x1p, y1p, x2p, y2p))

            out_name = f"{letter.upper()}.jpg"
            out_path = OUT_DIR / out_name
            crop.save(out_path, "JPEG", quality=95)

            print(f"Saved {out_name} -> {out_path}")

    print("\nDone! Check folder:", OUT_DIR)


if __name__ == "__main__":
    main()
