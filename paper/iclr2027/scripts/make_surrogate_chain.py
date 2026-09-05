#!/usr/bin/env python3
"""Generate the paper's draw-level paired-slope figure as a vector PDF."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]
SUMMARY = ROOT / "experiments" / "less_aligned" / "results_summary"
OUT = PAPER / "figures" / "surrogate_chain.pdf"


def esc(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


class PDF:
    def __init__(self, width: int = 1120, height: int = 460):
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def color(self, rgb: tuple[int, int, int]) -> str:
        return " ".join(f"{x / 255:.3f}" for x in rgb)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        rgb=(50, 50, 50),
        width=2,
        dash: str | None = None,
    ) -> None:
        d = f"[{dash}] 0 d" if dash else "[] 0 d"
        self.ops.append(
            f"q {self.color(rgb)} RG {width} w {d} "
            f"{x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S Q"
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill=(245, 245, 245),
        stroke=(90, 90, 90),
        radius: float = 0,
    ) -> None:
        # PDF has no primitive rounded rectangle; straight boxes print more
        # reliably and remain legible in grayscale.
        del radius
        self.ops.append(
            f"q {self.color(fill)} rg {self.color(stroke)} RG 1.5 w "
            f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re B Q"
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill=(255, 255, 255),
        stroke=(50, 50, 50),
        width=1.5,
    ) -> None:
        # Four cubic Bézier arcs approximate a circle.
        k = 0.5522847498 * radius
        self.ops.append(
            f"q {self.color(fill)} rg {self.color(stroke)} RG {width} w "
            f"{x + radius:.1f} {y:.1f} m "
            f"{x + radius:.1f} {y + k:.1f} {x + k:.1f} {y + radius:.1f} "
            f"{x:.1f} {y + radius:.1f} c "
            f"{x - k:.1f} {y + radius:.1f} {x - radius:.1f} {y + k:.1f} "
            f"{x - radius:.1f} {y:.1f} c "
            f"{x - radius:.1f} {y - k:.1f} {x - k:.1f} {y - radius:.1f} "
            f"{x:.1f} {y - radius:.1f} c "
            f"{x + k:.1f} {y - radius:.1f} {x + radius:.1f} {y - k:.1f} "
            f"{x + radius:.1f} {y:.1f} c B Q"
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 18,
        rgb=(30, 30, 30),
        bold=False,
        align="left",
    ) -> None:
        font = "/F2" if bold else "/F1"
        # Helvetica's average width is approximately 0.52 em; this is only for
        # centering short labels.
        if align == "center":
            x -= len(text) * size * 0.26
        elif align == "right":
            x -= len(text) * size * 0.52
        self.ops.append(
            f"BT {self.color(rgb)} rg {font} {size:.1f} Tf "
            f"1 0 0 1 {x:.1f} {y:.1f} Tm ({esc(text)}) Tj ET"
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        rgb=(65, 65, 65),
        width=2.5,
        dashed=False,
    ) -> None:
        self.line(x1, y1, x2, y2, rgb=rgb, width=width, dash="7 5" if dashed else None)
        # All arrows in this figure are horizontal.
        sign = 1 if x2 >= x1 else -1
        tip = 11
        self.line(x2, y2, x2 - sign * tip, y2 + 6, rgb=rgb, width=width)
        self.line(x2, y2, x2 - sign * tip, y2 - 6, rgb=rgb, width=width)

    def save(self, path: Path) -> None:
        content = "\n".join(self.ops).encode("latin-1")
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> "
                f"/Contents 4 0 R >>"
            ).encode()
        )
        objects.append(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        # Include a NUL in the binary marker so version-control tools classify
        # the generated asset as binary rather than line-oriented text.
        data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\x00\n")
        offsets = [0]
        for i, obj in enumerate(objects, 1):
            offsets.append(len(data))
            data.extend(f"{i} 0 obj\n".encode())
            data.extend(obj)
            data.extend(b"\nendobj\n")
        xref = len(data)
        data.extend(f"xref\n0 {len(objects)+1}\n".encode())
        data.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            data.extend(f"{off:010d} 00000 n \n".encode())
        data.extend(
            (
                f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n"
            ).encode()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def mean_cell(cells, prefix, draw, method, key):
    return sum(cells[f"{prefix}_draw{draw}_{method}_seed{s}"][key] for s in (42, 1)) / 2


def load_numbers():
    l2_geom = json.loads((SUMMARY / "bbh_forensic_geometry.json").read_text())
    l2_loss = json.loads((SUMMARY / "bbh_forensic_query_loss.json").read_text())
    l2_cot = json.loads((SUMMARY / "bbh_forensic_query_cot.json").read_text())
    l32 = json.loads((SUMMARY / "llama32_results.json").read_text())
    l32_diag = json.loads((SUMMARY / "llama32_diagnostics.json").read_text())
    out = {"Llama-2-7B": {}, "Llama-3.2-3B": {}}

    out["Llama-2-7B"]["d2"] = {
        method: [
            l2_geom["per_draw"][str(d)]["methods"][method]["D2_to_Q"]
            for d in range(3)
        ]
        for method in ("randk", "dsmc")
    }
    out["Llama-2-7B"]["query_ce"] = {
        method: [
            sum(
                l2_loss["cells"][f"bbhx_draw{d}_{method}_seed{s}"]["query_loss"]
                for s in (42, 1)
            )
            / 2
            for d in range(3)
        ]
        for method in ("randk", "dsmc")
    }
    out["Llama-2-7B"]["same_item"] = {
        method: [
            x * 100
            for x in l2_cot["methods"][method]["query_cot_em_per_draw"]
        ]
        for method in ("randk", "dsmc")
    }
    out["Llama-2-7B"]["heldout"] = {
        method: [
            l2_geom["per_draw"][str(d)]["methods"][method]["acc_seed_avg"] * 100
            for d in range(3)
        ]
        for method in ("randk", "dsmc")
    }

    out["Llama-3.2-3B"]["d2"] = {
        method: [l32["geometry"][str(d)]["D2"][method] for d in range(3)]
        for method in ("randk", "dsmc")
    }
    out["Llama-3.2-3B"]["query_ce"] = {
        method: [
            mean_cell(l32_diag["cells"], "l32", d, method, "wrapped_ce")
            for d in range(3)
        ]
        for method in ("randk", "dsmc")
    }
    out["Llama-3.2-3B"]["same_item"] = {
        method: [
            mean_cell(l32_diag["cells"], "l32", d, method, "cot_em") * 100
            for d in range(3)
        ]
        for method in ("randk", "dsmc")
    }
    out["Llama-3.2-3B"]["heldout"] = {
        method: [
            l32["per_draw"][str(d)][method]["draw_mean"] * 100
            for d in range(3)
        ]
        for method in ("randk", "dsmc")
    }
    return out


PANELS = (
    ("d2", "Target D2", "lower is better", lambda x: f"{x:.3f}"),
    ("query_ce", "Wrapped query CE", "lower is better", lambda x: f"{x:.1f}"),
    ("same_item", "Same-item EM", "higher is better", lambda x: f"{x:.0f}%"),
    ("heldout", "Held-out EM", "higher is better", lambda x: f"{x:.1f}%"),
)


def draw_panel(p, x, y, width, height, values, formatter, draw_colors):
    random_color = (218, 124, 48)
    dsmc_color = (48, 92, 151)
    grid = (210, 210, 210)
    axis = (95, 95, 95)
    x_rand = x + width * 0.34
    x_dsmc = x + width * 0.78
    y_bottom = y + 24
    y_top = y + height - 15

    all_values = values["randk"] + values["dsmc"]
    lo, hi = min(all_values), max(all_values)
    span = hi - lo
    pad = max(span * 0.18, abs(hi) * 0.015, 1e-4)
    lo, hi = lo - pad, hi + pad

    def sy(v):
        return y_bottom + (v - lo) / (hi - lo) * (y_top - y_bottom)

    # Axes, three horizontal guides, and y tick labels.
    p.line(x + 42, y_bottom, x + 42, y_top, rgb=axis, width=1)
    p.line(x + 42, y_bottom, x + width - 8, y_bottom, rgb=axis, width=1)
    for frac in (0.0, 0.5, 1.0):
        yy = y_bottom + frac * (y_top - y_bottom)
        p.line(x + 42, yy, x + width - 8, yy, rgb=grid, width=0.7)
        p.text(x + 37, yy - 3, formatter(lo + frac * (hi - lo)), 9, rgb=axis, align="right")

    for d, color in enumerate(draw_colors):
        yr = sy(values["randk"][d])
        yd = sy(values["dsmc"][d])
        p.line(x_rand, yr, x_dsmc, yd, rgb=color, width=2.2)
        p.circle(x_rand, yr, 4.5, fill=random_color, stroke=color, width=1.2)
        p.circle(x_dsmc, yd, 4.5, fill=dsmc_color, stroke=color, width=1.2)

    p.text(x_rand, y + 7, "Random", 9.5, rgb=random_color, bold=True, align="center")
    p.text(x_dsmc, y + 7, "DSMC", 9.5, rgb=dsmc_color, bold=True, align="center")


def main() -> None:
    n = load_numbers()
    p = PDF()
    blue = (48, 92, 151)
    orange = (218, 124, 48)
    gray = (92, 92, 92)
    draw_colors = ((74, 85, 104), (116, 126, 145), (158, 166, 181))

    p.text(20, 435, "BBH paired Random-to-DSMC changes across three query draws", 20, bold=True)
    p.text(20, 414, "Each panel uses its own y-scale; lines connect the same draw.", 11.5, rgb=gray)
    p.circle(760, 421, 4, fill=orange, stroke=orange)
    p.text(770, 417, "Random", 10, rgb=orange)
    p.circle(835, 421, 4, fill=blue, stroke=blue)
    p.text(845, 417, "DSMC", 10, rgb=blue)
    for d, color in enumerate(draw_colors):
        xx = 915 + d * 58
        p.line(xx, 421, xx + 18, 421, rgb=color, width=2.2)
        p.text(xx + 23, 417, f"D{d}", 9.5, rgb=color)

    panel_x = (65, 330, 595, 860)
    row_y = {"Llama-2-7B": 225, "Llama-3.2-3B": 30}
    panel_w, panel_h = 245, 150

    for j, (_key, title, direction, _formatter) in enumerate(PANELS):
        p.text(panel_x[j] + panel_w / 2, 394, title, 13, bold=True, align="center")
        p.text(panel_x[j] + panel_w / 2, 379, direction, 9.5, rgb=gray, align="center")

    for stack, y in row_y.items():
        p.text(8, y + 78, stack, 11.5, rgb=blue, bold=True)
        for j, (key, _title, _direction, formatter) in enumerate(PANELS):
            draw_panel(
                p,
                panel_x[j],
                y,
                panel_w,
                panel_h,
                n[stack][key],
                formatter,
                draw_colors,
            )

    p.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
