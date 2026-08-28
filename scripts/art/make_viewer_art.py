"""Turn the nano-banana source renders into the viewer's art assets.

Sources (committed, see docs/plans/2026-08-27-gnomic-design.md):
- scripts/art/source/gnomes_sheet.png — Ivan, Anton, Yura on a flat green
  chroma backdrop, generated with gemini-2.5-flash-image anchored on
  coworld-heartleaf/docs/heartleafBanner.png.
- scripts/art/source/moot_hero.png — the Moot at the village tree plaza.

Outputs (committed, served by the game pod at /client/art/*):
- gnomic/server/viewer/art/{ivan,anton,yura}.png — 160px keyed bust chips.
- gnomic/server/viewer/art/moot_hero.png — 1200px-wide hero.

Run: uv run --with pillow python scripts/art/make_viewer_art.py
"""

from collections import Counter, deque
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "art" / "source"
OUT = ROOT / "gnomic" / "server" / "viewer" / "art"
GNOMES = ["ivan", "anton", "yura"]
TOLERANCE = 60


def backdrop_color(im: Image.Image) -> tuple[int, int, int]:
    px = im.load()
    w, h = im.size
    border = [px[x, 0] for x in range(w)] + [px[x, h - 1] for x in range(w)]
    border += [px[0, y] for y in range(h)] + [px[w - 1, y] for y in range(h)]
    return Counter(p[:3] for p in border).most_common(1)[0][0]


def key_out(im: Image.Image) -> Image.Image:
    """Flood-fill the chroma backdrop from the border to transparency."""
    im = im.convert("RGBA")
    px = im.load()
    w, h = im.size
    bg = backdrop_color(im)

    def is_bg(p: tuple[int, ...]) -> bool:
        return sum(abs(a - b) for a, b in zip(p[:3], bg)) <= TOLERANCE

    seen = [[False] * h for _ in range(w)]
    queue: deque[tuple[int, int]] = deque()
    for x in range(w):
        queue.extend(((x, 0), (x, h - 1)))
    for y in range(h):
        queue.extend(((0, y), (w - 1, y)))
    while queue:
        x, y = queue.popleft()
        if not (0 <= x < w and 0 <= y < h) or seen[x][y]:
            continue
        seen[x][y] = True
        if not is_bg(px[x, y]):
            continue
        px[x, y] = (0, 0, 0, 0)
        queue.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
    return im


def split_columns(im: Image.Image) -> list[Image.Image]:
    """Split the keyed row on fully transparent columns."""
    w, h = im.size
    alpha = im.getchannel("A").load()
    occupied = [any(alpha[x, y] > 8 for y in range(h)) for x in range(w)]
    spans, start = [], None
    for x, filled in enumerate(occupied + [False]):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            if x - start > w // 12:
                spans.append((start, x))
            start = None
    if len(spans) != len(GNOMES):
        raise SystemExit(f"expected {len(GNOMES)} figures, found {len(spans)}: {spans}")
    return [im.crop((a, 0, b, h)) for a, b in spans]


def bust(chip: Image.Image, size: int = 160) -> Image.Image:
    """Trim to content, then take the top square (cap + face + shoulders)."""
    box = chip.getbbox()
    chip = chip.crop(box)
    w, h = chip.size
    side = min(w, int(w * 1.05), h)
    top = chip.crop(((w - side) // 2, 0, (w + side) // 2, side))
    return top.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sheet = key_out(Image.open(SOURCE / "gnomes_sheet.png"))
    for name, chip in zip(GNOMES, split_columns(sheet)):
        bust(chip).save(OUT / f"{name}.png", optimize=True)
        print(f"wrote {OUT / f'{name}.png'}")
    hero = Image.open(SOURCE / "moot_hero.png").convert("RGB")
    ratio = 1200 / hero.width
    hero = hero.resize((1200, round(hero.height * ratio)), Image.LANCZOS)
    hero.save(OUT / "moot_hero.png", optimize=True)
    print(f"wrote {OUT / 'moot_hero.png'}")


if __name__ == "__main__":
    main()
