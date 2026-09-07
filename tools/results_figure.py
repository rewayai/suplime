#!/usr/bin/env python
"""Draw assets/results{,_dark}.svg + results.png from the macro-average rows of README.md.

    python tools/results_figure.py            # after editing the README tables

Two panels of horizontal bars (lower DER is better): pyannote's 8-corpus benchmark
(four systems) and the full 12-corpus set (the two systems evaluated on it).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

R = Path(__file__).resolve().parents[1]
SYSTEMS = ["SUPlime", "DiariZen-L-s80-v2", "pyannoteAI precision-2", "pyannote community-1"]


def read_macros(readme: Path):
    s = readme.read_text()
    out = {}
    for n in (8, 12):
        m = re.search(r"\| \*\*macro average \(%d\)\*\* \|(.*)" % n, s)
        cells = [c.strip().strip("*").replace("†", "").strip() for c in m.group(1).strip().strip("|").split("|")]
        out[n] = {sysname: float(c) for sysname, c in zip(SYSTEMS, cells) if c and c != "—"}
    return out


THEMES = {
    "light": dict(surface="#ffffff", text="#0b0b0b", text2="#52514e", muted="#898781", accent="#4f8f08", other="#a3a19a"),
    "dark": dict(surface="#0d1117", text="#ffffff", text2="#c3c2b7", muted="#8b949e", accent="#8fd130", other="#7a7975"),
}


def panel(ax, title, values, t, xmax, slots, fig):
    # SUPlime first, then the others best (lowest) first
    order = ["SUPlime"] + sorted((k for k in values if k != "SUPlime"), key=values.get)
    best = min(values, key=values.get)
    ax.set_facecolor(t["surface"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, slots - 0.4)  # same slot count in every panel = same bar thickness
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=11, color=t["text"])
    ax.tick_params(axis="y", length=0, pad=10)
    # pixel geometry, for a 4px rounded data-end regardless of the data scale
    bb = ax.get_position()
    w_px = bb.width * fig.get_figwidth() * fig.dpi
    h_px = bb.height * fig.get_figheight() * fig.dpi
    x_per_px = xmax / w_px
    y_per_px = slots / h_px
    h = 22 * y_per_px  # 22 px thick
    r = 4 * x_per_px
    for i, name in enumerate(order):
        v = values[name]
        color = t["accent"] if name == "SUPlime" else t["other"]
        ax.add_patch(FancyBboxPatch((0, i - h / 2), v, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    mutation_aspect=y_per_px / x_per_px, linewidth=0, facecolor=color))
        ax.add_patch(Rectangle((0, i - h / 2), min(2 * r, v), h, linewidth=0, facecolor=color))  # square baseline end
        ax.text(v + 8 * x_per_px, i, f"{v:.2f}", va="center", ha="left", fontsize=11,
                color=t["text"], fontweight="bold" if name == best else "normal")  # bold = best
    ax.axvline(0, color=t["muted"], linewidth=1)
    ax.set_title(title, loc="left", fontsize=11.5, color=t["text2"], pad=10)


def draw(macros, theme: str, out: Path):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), dpi=100, gridspec_kw={"wspace": 0.75})
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.215, right=0.965, top=0.76, bottom=0.2)
    xmax = max(max(macros[8].values()), max(macros[12].values())) * 1.2
    slots = max(len(macros[8]), len(macros[12]))
    panel(axes[0], "pyannote 8-corpus benchmark", macros[8], t, xmax, slots, fig)
    panel(axes[1], "all 12 corpora", macros[12], t, xmax, slots, fig)
    fig.suptitle("Diarization error rate, macro average (%). Lower is better.", x=0.02, y=0.975,
                 ha="left", fontsize=13, color=t["text"], fontweight="bold")
    fig.text(0.02, 0.035, "Collar 0 s, overlapped speech scored, no oracle speaker count, one clustering threshold for all corpora.\n"
             "DiariZen, community-1: the authors' published numbers where available, otherwise our re-run of their open-source pipeline.",
             fontsize=8.2, color=t["muted"], ha="left", va="bottom", linespacing=1.5)
    fig.savefig(out, facecolor=t["surface"], dpi=200 if out.suffix == ".png" else 100)
    plt.close(fig)


def main():
    macros = read_macros(R / "README.md")
    assets = R / "assets"
    assets.mkdir(exist_ok=True)
    draw(macros, "light", assets / "results.svg")
    draw(macros, "light", assets / "results.png")
    draw(macros, "dark", assets / "results_dark.svg")
    print({n: macros[n] for n in macros})


if __name__ == "__main__":
    main()
