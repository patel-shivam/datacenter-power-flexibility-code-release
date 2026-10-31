"""Shared chart style. One place so every figure in this suite reads as one system.

Palette
-------
Values come from a validated reference palette and were re-checked with its own
validator rather than eyeballed:

  * **Ordinal ramp (model size).** Model size is an *ordered* dimension, so it gets a
    single-hue light-to-dark ramp, not categorical hues. The validator caps this at
    **five** steps: nine steps of the blue ramp fail the adjacent-lightness gate
    (gaps of 0.047 against a 0.06 floor), which is the general "more than ~7 colour
    classes blur" rule showing up concretely. Nine models therefore go in **small
    multiples**, and the ramp is used only when at most five series share an axis.
  * **Categorical slots** for genuinely nominal comparisons (pack vs spread, target vs
    measured). Slots 1-3 pass all-pairs CVD and normal-vision separation; slot 3
    (aqua) sits below 3:1 on the light surface, so any chart using three series
    carries direct labels as relief.

Rules this module enforces so individual charts cannot forget them
------------------------------------------------------------------
* **Never a dual axis.** Two measures of different scale get two panels. This is the
  single most misleading chart form -- the alignment of the two scales is arbitrary,
  so the picture invents a correlation the data does not contain. Every figure here
  that shows power *and* throughput shows them as separate panels.
* Hairline **solid** grid (dashes add noise and read as uncertainty), recessive axes,
  no top/right spines.
* Thin marks: 1.6 pt lines, ~9 px markers.
* Text wears ink tokens, never a series colour.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- ink / chrome -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- categorical slots (validated all-pairs, light) ---------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
       "#008300", "#4a3aa7", "#e34948"]

# --- ordinal ramp: at most FIVE steps (see module docstring) ------------------
_ORDINAL_5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MAX_ORDINAL = len(_ORDINAL_5)

# --- the card a figure's watts belong to --------------------------------------
# Name, and the one fact about its power that changes how a figure reads. Measured on our
# cards, not spec sheets: H100 from concurrency_sweep P(b) (warm idle ~115 W, the first request jumps to
# 290-470 W, saturating ~300-650 W by model); RTX 8000 from the 8-GPU fleet fit
# P = 153 + 225 W x lit and concurrency_sweep, where P(b) is flat above b = 1.
HARDWARE = {
    "h100-80gb": ("NVIDIA H100 80GB",
                  "per-replica power rises with batch: ~115 W warm idle, ~290–650 W busy"),
    "rtx-8000-48gb": ("Quadro RTX 8000 48GB (sm_75, fp16)",
                      "per-replica power is ~on/off: ~19 W dark, ~200–260 W lit at any batch"),
    # the H100's two-part shape at lower power, peaking at b ~ 64-96
    "v100-32gb": ("Tesla V100 32GB (sm_70, fp16)",
                  "per-replica power rises with batch: ~62–65 W warm idle, ~160–210 W at one "
                  "sequence, 200–276 W at b ≈ 64–96"),
    # bf16: ~55-61 W warm idle, ~400-500 W at one sequence, the 600 W limit from b ~ 96
    "rtx-5090-32gb": ("GeForce RTX 5090 32GB (sm_120, bf16)",
                      "per-replica power: ~55–61 W warm idle, 400–500 W at one sequence, "
                      "the 600 W limit from b ≈ 96"),
}


def hardware_label(slug: str) -> tuple[str, str]:
    """(display name, power characteristic) for a hardware slug; unknown -> (slug, '')."""
    return HARDWARE.get(slug, (slug, ""))


def ordinal_colors(n: int) -> list[str]:
    """``n`` steps of the single-hue ramp, light to dark. ``n`` must be <= 5.

    Refuses larger n rather than interpolating: the validator rejects six steps of
    this ramp on adjacent lightness, so a generated sixth colour would be a step the
    reader cannot separate from its neighbour.
    """
    if n > MAX_ORDINAL:
        raise ValueError(
            f"ordinal ramp is capped at {MAX_ORDINAL} steps (validator fails at 6); "
            f"asked for {n}. Use small multiples or select representative series.")
    if n <= 1:
        return [_ORDINAL_5[2]]
    idx = [round(i * (MAX_ORDINAL - 1) / (n - 1)) for i in range(n)]
    return [_ORDINAL_5[i] for i in idx]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # never dashed
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_2,
        "font.size": 9,
        "figure.dpi": 140,
    })


def finish(ax, *, xlabel=None, ylabel=None, title=None, legend=False,
           logx=False, logy=False) -> None:
    """Consistent per-axes finishing. Legend only when there are >= 2 series."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) >= 2:
            ax.legend(loc="best")


def label_points(ax, xs, ys, labels, *, dx=0, dy=9, fontsize=7.5, color=INK_2,
                 stagger=True):
    """Direct labels, with collision avoidance for points that crowd together.

    Crowding is the normal case here, not the exception: three of the nine models sit
    between 7.2 and 8.2 B, so on a log-parameter axis their labels land on top of each
    other and the chart becomes unreadable exactly where it is most interesting.
    Points whose neighbour is within 9% of the axis range in x get their label pushed
    to the next of three vertical levels.

    **Call this after the axis scale is set.** Crowding is measured in the axis's own
    coordinates, and ``ax.get_xscale()`` is read here -- labelling before
    ``set_xscale("log")`` measures gaps linearly on what will be a log axis, which gets
    it wrong at both ends (it would call 1.2 B / 1.7 B crowded when log-space puts them far
    apart, and 14.8 B / 23.6 B uncrowded when log-space puts them together).
    """
    import numpy as _np

    xs = _np.asarray(xs, float)
    ys = _np.asarray(ys, float)
    # Nothing to label is a normal case (e.g. no model turns down on a workload), and
    # nanmax of an empty array raises.
    if xs.size == 0:
        return
    order = _np.argsort(xs)
    xr = (_np.nanmax(xs) - _np.nanmin(xs)) or 1.0
    if ax.get_xscale() == "log":
        lx = _np.log10(_np.where(xs > 0, xs, _np.nan))
        xr = (_np.nanmax(lx) - _np.nanmin(lx)) or 1.0
        pos = lx
    else:
        pos = xs

    # Three offset levels, not two: the 7.2 / 8.0 / 8.2 B cluster is a run of THREE
    # crowded points, and a two-level alternation puts the first and third back on
    # the same line.
    levels = [dy, -(dy + 10), dy + 15]   # >= ~14 px apart; text is ~10 px tall
    prev = None
    run = 0
    offsets = {}
    for i in order:
        crowded = stagger and prev is not None and abs(pos[i] - prev) < 0.09 * xr
        run = (run + 1) if crowded else 0
        offsets[i] = levels[run % len(levels)]
        prev = pos[i]

    for i in range(len(xs)):
        ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx, offsets[i]), ha="center", fontsize=fontsize,
                    color=color, annotation_clip=False)


def save(fig, path, *, tight=True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
