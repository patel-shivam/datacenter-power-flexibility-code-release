"""Shared style, paths, table loaders and helpers for the paper's figures.

Every figure module imports this. The style is the one the paper's figures were drawn
with: the suite's base chart style (``analyze/plotting.py``) with pure white surfaces,
serif text with Computer Modern mathtext, and black labels on the single-column figures.

Paths are resolved from this file, so the figures can be regenerated from any working
directory: ``python make_figures.py`` at the repository's root.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent          # the repository's root directory
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analyze"))
import plotting as P  # noqa: E402  (analyze/plotting.py: the suite's palette and base style; selects Agg)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 120)

RESULTS = ROOT / "results"
DATA_SG = RESULTS / "tables" / "single_gpu"       # tables written by analyze/plot_single_gpu_model.py
DATA_FR = RESULTS / "tables" / "fleet_routing"    # tables written by analyze/plot_fleet_routing_figures.py
DATA_DC = RESULTS / "tables" / "datacenter"               # tables written by analyze/datacenter_sim.py
FIG_ROOT = ROOT / "figures"                        # make_figures.py writes figures/main and figures/appendix

# ---- core models and workloads ----
MODELS = ["llama32-1b", "qwen3-4b", "llama31-8b", "qwen3-8b"]
TASKS = ["fixed_256_256", "fixed_512_512", "fixed_1024_1024",
         "sharegpt", "humaneval", "arxiv_summarization"]

# Paul Tol vibrant palette; additional colours for non-core categories.
SCI6 = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#EE3377", "#33BBEE"]
EXTRA_COLORS = ["#777777", "#999933", "#332288"]

# ---- STYLE: shared colours, markers, labels, sizes (task = colour, model = marker) ----
STYLE = dict(
    wl_order=TASKS + ["fixed_128_128", "fixed_2048_2048", "fixed_6144_1024"],
    wl_label={
        "sharegpt": "ShareGPT",
        "humaneval": "HumanEval",
        "fixed_128_128": "128 / 128",
        "fixed_256_256": "256 / 256",
        "fixed_512_512": "512 / 512",
        "fixed_1024_1024": "1024 / 1024",
        "fixed_2048_2048": "2048 / 2048",
        "fixed_6144_1024": "6144 / 1024",
        "arxiv_summarization": "arXiv summ.",
    },
    model_order=[
        "llama32-1b", "qwen3-4b", "llama31-8b", "qwen3-8b",
        "qwen3-1.7b", "qwen3-14b", "llama32-3b", "mistral-7b", "mistral-24b",
    ],
    model_label={
        "llama32-1b": "Llama-3.2-1B",
        "qwen3-4b": "Qwen3-4B",
        "llama31-8b": "Llama-3.1-8B",
        "qwen3-8b": "Qwen3-8B",
        "qwen3-1.7b": "Qwen3-1.7B",
        "qwen3-14b": "Qwen3-14B",
        "llama32-3b": "Llama-3.2-3B",
        "mistral-7b": "Mistral-7B",
        "mistral-24b": "Mistral-24B",
    },
    hw_order=["h100-80gb", "rtx-8000-48gb", "v100-32gb", "rtx-5090-32gb"],
    hw_label={
        "h100-80gb": "H100 80 GB",
        "rtx-8000-48gb": "RTX 8000 48 GB (fp16)",
        "v100-32gb": "V100 32 GB (fp16)",
        "rtx-5090-32gb": "RTX 5090 32 GB",
    },
    core_models=MODELS.copy(),
    core_tasks=TASKS.copy(),
    pair_mark={"llama31-8b": "^", "qwen3-8b": "D"},
    pair_label={
        "llama31-8b": "Llama-3.1-8B ×3 + Llama-3.2-1B",
        "qwen3-8b": "Qwen3-8B ×3 + Qwen3-4B",
    },
    eight_mark={(5, 3): "s", (4, 4): "p", (6, 2): "h"},
    band=(0.8, 1.25),
    parity_size=28,
    load_cmap="viridis_r",
    formats=("png", "pdf"),
    panel_w=3.9,
    panel_h=4.6,
)
STYLE["wl_color"] = dict(zip(STYLE["wl_order"], SCI6 + EXTRA_COLORS))
STYLE["model_color"] = dict(zip(STYLE["model_order"], SCI6 + EXTRA_COLORS))
STYLE["model_mark"] = dict(zip(STYLE["model_order"], ["o", "s", "^", "D", "v", "P", "X", "*", "p"]))

WL, WLL, WLC = STYLE["wl_order"], STYLE["wl_label"], STYLE["wl_color"]
MM, ML, MC = STYLE["model_mark"], STYLE["model_label"], STYLE["model_color"]
HWL = STYLE["hw_label"]

# Paper figures: PURE WHITE surfaces (the suite's base style tints them off-white).
SURFACE = "#ffffff"

# Serif text with Computer Modern mathtext, embedded (Type 42) fonts in the PDFs.
CFG_MATHTEXT_CM = {
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
CFG_SIZES = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    """The figure style, applied in the order the paper's figures were drawn with."""
    P.apply_style()
    P.SURFACE = SURFACE
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "savefig.edgecolor": "none", "legend.facecolor": SURFACE})
    mpl.rcParams.update(CFG_MATHTEXT_CM)
    mpl.rcParams.update(CFG_SIZES)


# ---- output -------------------------------------------------------------------------------
WRITTEN: list[dict] = []        # every figure written in this process: placement, name, files


def set_output(root: Path) -> None:
    global FIG_ROOT
    FIG_ROOT = Path(root)


def savefig(fig, name: str, placement: str) -> None:
    """Write FIG_ROOT/<placement>/<name>.<fmt> for every format in STYLE and record it."""
    out = FIG_ROOT / placement
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in STYLE["formats"]:
        path = out / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        files.append(path)
    plt.close(fig)
    WRITTEN.append({"placement": placement, "name": name, "files": files})
    print(f"  wrote {placement}/{name}.{{{','.join(STYLE['formats'])}}}")


# ---- helpers ---------------------------------------------------------------------------------
def boolcol(df: pd.DataFrame, col: str) -> pd.Series:
    """A CSV boolean column (True/False strings, blanks for absent keys) as bool."""
    if col not in df:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(("true", "1", "1.0"))


def stats(x, y) -> dict:
    """measured/predicted statistics: n, median ratio, p10-p90, MAPE, share within 10/20 %."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if not len(x):
        return {"n": 0}
    r, ape = y / x, np.abs(x - y) / y
    return {"n": int(len(x)), "median_ratio": float(np.median(r)), "p10_ratio": float(np.percentile(r, 10)),
            "p90_ratio": float(np.percentile(r, 90)), "mape_pct": float(100 * ape.mean()),
            "within_10pct": float(np.mean(ape <= 0.10)), "within_20pct": float(np.mean(ape <= 0.20))}


def black(ax, title=9, label=9, tick=8):
    """The main figures' finish: black labels, ticks and spines. (``ax.title`` is the CENTRE
    title; the panel titles set through P.finish are left titles and keep the style's size.)"""
    ax.title.set(fontsize=title, color="black")
    ax.xaxis.label.set(fontsize=label, color="black")
    ax.yaxis.label.set(fontsize=label, color="black")
    ax.tick_params(axis="both", which="both", labelsize=tick, colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    for axis in (ax.xaxis, ax.yaxis):
        axis.get_offset_text().set_color("black")


# ---- table loaders ---------------------------------------------------------------------------
def load_single_gpu() -> SimpleNamespace:
    """The single-GPU data layer (analyze/plot_single_gpu_model.py)."""
    T = SimpleNamespace(
        pts=pd.read_csv(DATA_SG / "single_gpu_points.csv"),          # every scored point (x predicted, y measured)
        lv=pd.read_csv(DATA_SG / "single_gpu_levels.csv"),           # every closed-loop level of every cell
        pbc=pd.read_csv(DATA_SG / "single_gpu_pb_curves.csv"),       # fitted P(b) sampled on a b grid
        cal=pd.read_csv(DATA_SG / "single_gpu_calibration.csv"),     # t0, t1, R2, P(b) form per cell
    )
    for df in (T.pts, T.lv):
        for c in ("hollow", "kv_bound", "clamped", "offline", "peak", "short", "valid", "degenerate"):
            if c in df:
                df[c] = boolcol(df, c)
    print(f"single GPU: {len(T.cal)} cells on {T.cal.hw.nunique()} GPUs, {len(T.lv)} levels, {len(T.pts)} scored points")
    return T


def load_fleet() -> SimpleNamespace:
    """The fleet-routing data layer (analyze/plot_fleet_routing_figures.py)."""
    F = SimpleNamespace(
        meta=pd.read_csv(DATA_FR / "fleet_meta.csv").set_index("fleet"),
        probes=pd.read_csv(DATA_FR / "fleet_probes.csv"),
        curves=pd.read_csv(DATA_FR / "fleet_model_curves.csv"),
        grid=pd.read_csv(DATA_FR / "fleet_model_band_grid.csv"),
        bandm=pd.read_csv(DATA_FR / "fleet_band_measured.csv"),
        winb=pd.read_csv(DATA_FR / "fleet_window_band.csv"),
    )
    for c in ("feasible", "wide"):
        F.probes[c] = boolcol(F.probes, c)
    F.curves["in_window"] = boolcol(F.curves, "in_window")
    F.curves["wide"] = boolcol(F.curves, "wide")
    print(f"fleets: {len(F.meta)} ({', '.join(F.meta.index)}); held probes {len(F.probes)}, model-curve points {len(F.curves)}")
    return F


def read_dc(name: str, **kw) -> pd.DataFrame:
    """One table of the datacenter simulation."""
    return pd.read_csv(DATA_DC / name, **kw)
