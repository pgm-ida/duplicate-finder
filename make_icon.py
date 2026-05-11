#!/usr/bin/env python3
"""Render the Duplicate "ditto" mark (two pills) into icon.ico.

Each ICO sub-image is rendered fresh at its target size rather than
downsampled from a single high-res source, so the 16/24/32 px favicon
sizes stay crisp instead of going muddy.

Geometry mirrors the SVG in logos.jsx (DittoGlyph): two rounded pills
horizontally centered on a rounded-square cream background.
"""

from PIL import Image, ImageDraw

CREAM = (243, 240, 232, 255)
INK = (17, 17, 17, 255)


def render_ditto(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Background: cream rounded square. Looks correct on both light and
    # dark host shells (Windows taskbar, file explorer).
    corner = max(2, int(size * 0.22))
    d.rounded_rectangle([(0, 0), (size - 1, size - 1)], radius=corner, fill=CREAM)

    # Two pills, beefed slightly compared to the SVG so they still read
    # at 16 px after antialiasing.
    pill_w = size * 0.17
    pill_h = size * 0.46
    pill_r = pill_w / 2
    gap = size * 0.13
    total_w = pill_w * 2 + gap
    x0 = (size - total_w) / 2
    y0 = (size - pill_h) / 2

    d.rounded_rectangle(
        [(x0, y0), (x0 + pill_w, y0 + pill_h)],
        radius=pill_r, fill=INK,
    )
    x1 = x0 + pill_w + gap
    d.rounded_rectangle(
        [(x1, y0), (x1 + pill_w, y0 + pill_h)],
        radius=pill_r, fill=INK,
    )
    return img


def make_ico(path: str = "icon.ico") -> None:
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = [render_ditto(s) for s in sizes]
    # Pillow's ICO writer takes one base image + append_images for the
    # rest. `sizes` tells it which sub-images to keep — it picks the
    # closest source for each.
    imgs[-1].save(
        path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=imgs[:-1],
    )


if __name__ == "__main__":
    make_ico("icon.ico")
    print("Wrote icon.ico")
