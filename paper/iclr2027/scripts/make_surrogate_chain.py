#!/usr/bin/env python3
"""Generate the paper's conceptual/result figure as a dependency-free PDF."""

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
    def __init__(self, width: int = 1000, height: int = 400):
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

        data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
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


def load_numbers():
    l2_geom = json.loads((SUMMARY / "bbh_forensic_geometry.json").read_text())
    l2_loss = json.loads((SUMMARY / "bbh_forensic_query_loss.json").read_text())
    l2_cot = json.loads((SUMMARY / "bbh_forensic_query_cot.json").read_text())
    l32 = json.loads((SUMMARY / "llama32_results.json").read_text())
    l32_diag = json.loads((SUMMARY / "llama32_diagnostics.json").read_text())
    return {
        "l2": {
            "d2": (
                l2_geom["pooled_secondary"]["per_method"]["dsmc"]["D2_mean"]
                - l2_geom["pooled_secondary"]["per_method"]["randk"]["D2_mean"]
            ),
            "ce": (
                l2_loss["method_summary"]["dsmc"]["delta_query_loss_mean"]
                - l2_loss["method_summary"]["randk"]["delta_query_loss_mean"]
            ),
            "qem": (
                l2_cot["methods"]["dsmc"]["delta_query_cot_em_vs_base"]
                - l2_cot["methods"]["randk"]["delta_query_cot_em_vs_base"]
            )
            * 100,
            "hem": (
                l2_cot["methods"]["dsmc"]["heldout_em_mean"]
                - l2_cot["methods"]["randk"]["heldout_em_mean"]
            )
            * 100,
        },
        "l32": {
            "d2": sum(
                x["D2"]["dsmc"] - x["D2"]["randk"]
                for x in l32["geometry"].values()
            )
            / 3,
            "ce": (
                l32_diag["method_summary"]["dsmc"]["d_wrapped_ce_mean"]
                - l32_diag["method_summary"]["randk"]["d_wrapped_ce_mean"]
            ),
            "qem": (
                l32_diag["method_summary"]["dsmc"]["d_cot_em_mean"]
                - l32_diag["method_summary"]["randk"]["d_cot_em_mean"]
            )
            * 100,
            "hem": (
                l32["methods"]["dsmc"]["mean_over_draws"]
                - l32["methods"]["randk"]["mean_over_draws"]
            )
            * 100,
        },
    }


def main() -> None:
    n = load_numbers()
    p = PDF()
    blue = (48, 92, 151)
    green = (74, 145, 84)
    orange = (218, 124, 48)
    red = (174, 54, 52)
    gray = (92, 92, 92)

    p.text(35, 365, "BBH: three-draw mean DSMC - Random contrasts", 23, bold=True)
    p.text(
        35,
        335,
        "Lower is better for D2 and targeting loss; higher is better for exact match.",
        16,
        rgb=gray,
    )
    x_positions = [235, 440, 640, 835]
    headings = ["Target D2", "Targeting loss", "Same-item EM", "Held-out EM"]
    for x, h in zip(x_positions, headings):
        p.text(x, 290, h, 16, rgb=gray, bold=True, align="center")

    rows = [
        ("Llama-2-7B", n["l2"], blue),
        ("Llama-3.2-3B", n["l32"], green),
    ]
    ys = [205, 115]
    for (name, vals, color), y in zip(rows, ys):
        p.text(30, y + 10, name, 17, rgb=color, bold=True)
        p.rect(150, y - 10, 170, 52, fill=(239, 248, 239), stroke=green)
        p.text(
            235,
            y + 11,
            f"{vals['d2']:+.3f}",
            16,
            rgb=green,
            bold=True,
            align="center",
        )
        p.rect(355, y - 10, 170, 52, fill=(239, 248, 239), stroke=green)
        p.text(440, y + 11, f"{vals['ce']:+.2f} nats", 16, rgb=green, bold=True, align="center")
        p.rect(555, y - 10, 170, 52, fill=(253, 239, 237), stroke=red)
        p.text(640, y + 11, f"{vals['qem']:+.2f} pp", 16, rgb=red, bold=True, align="center")
        p.rect(750, y - 10, 170, 52, fill=(253, 239, 237), stroke=red)
        p.text(835, y + 11, f"{vals['hem']:+.2f} pp", 16, rgb=red, bold=True, align="center")
        p.arrow(320, y + 16, 350, y + 16, rgb=orange)
        p.arrow(525, y + 16, 550, y + 16, rgb=red, dashed=True)
        p.arrow(725, y + 16, 745, y + 16, rgb=red, dashed=True)

    p.text(
        35,
        45,
        "DSMC is closer in geometry and targeting loss, but worse in both task metrics.",
        18,
        rgb=red,
        bold=True,
    )
    p.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
