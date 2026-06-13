"""Generate the Timbre Studio app icon (1024x1024 PNG) with the stdlib only.

Dark warm rounded square plus five amber waveform capsules, matching the in-app
brand glyph. Output: studio/icon-src.png, then feed it to
`npm run tauri icon icon-src.png`.
"""
import struct
import zlib
from pathlib import Path

SIZE = 1024
RADIUS = 190

INK = (20, 17, 11)        # #14110B
INK_TOP = (34, 28, 18)    # subtle vertical gradient top
AMBER = (242, 163, 60)    # #F2A33C
AMBER_HOT = (255, 196, 107)

# capsule bars: (center_x, half_height), a waveform glyph
BARS = [
    (252, 110),
    (382, 230),
    (512, 360),
    (642, 180),
    (772, 110),
]
BAR_HALF_W = 34


def rounded_sq_alpha(x: float, y: float) -> float:
    """1.0 inside the rounded square, 0 outside, soft 1.5px edge."""
    inset = 28
    lo, hi = inset, SIZE - inset
    cx = min(max(x, lo + RADIUS), hi - RADIUS)
    cy = min(max(y, lo + RADIUS), hi - RADIUS)
    d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 - RADIUS
    if x >= lo + RADIUS and x <= hi - RADIUS:
        d = max(lo - y, y - hi)
    elif y >= lo + RADIUS and y <= hi - RADIUS:
        d = max(lo - x, x - hi)
    return max(0.0, min(1.0, 0.5 - d / 1.5 + 0.5))


def capsule_alpha(x: float, y: float) -> float:
    cy = SIZE / 2
    best = 1e9
    for bx, hh in BARS:
        # distance to a vertical capsule centered at (bx, cy)
        py = min(max(y, cy - hh), cy + hh)
        d = ((x - bx) ** 2 + (y - py) ** 2) ** 0.5 - BAR_HALF_W
        best = min(best, d)
    return max(0.0, min(1.0, 0.5 - best / 1.5 + 0.5))


def lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make_rows():
    rows = bytearray()
    for y in range(SIZE):
        rows.append(0)  # filter: none
        ty = y / SIZE
        bg = lerp(INK_TOP, INK, min(1.0, ty * 1.4))
        for x in range(SIZE):
            sq = rounded_sq_alpha(x + 0.5, y + 0.5)
            if sq <= 0:
                rows += b"\x00\x00\x00\x00"
                continue
            cap = capsule_alpha(x + 0.5, y + 0.5)
            if cap > 0:
                bar = lerp(AMBER, AMBER_HOT, max(0.0, 1 - (abs(y - SIZE / 2) / 400)))
                col = lerp(bg, bar, cap)
            else:
                col = bg
            rows += bytes(col) + bytes([round(255 * sq)])
    return bytes(rows)


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def main():
    ihdr = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)
    idat = zlib.compress(make_rows(), 9)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", idat)
        + png_chunk(b"IEND", b"")
    )
    out = Path(__file__).resolve().parent.parent / "icon-src.png"
    out.write_bytes(png)
    print(f"wrote {out} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
