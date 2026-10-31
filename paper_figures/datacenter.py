"""Datacenter-scale figures: the analytical fleet model scaled to a 10,000-GPU site and driven
by public data (a simulation, not a measurement; analyze/datacenter_sim.py).

Three figures, each drawn for three sites:
  week_flexibility    a week of real traffic: facility power in normal operation and how far
                      routing can lower it, at every share of the queries flexible (10-100 %)
  event_day           a 4-hour demand-response event on the busiest day, with the first and
                      last minute at 1 s resolution
  gpus_leaving        the power reduction as inference GPUs leave (lent to training, failed, in
                      maintenance), and a day with 40 % of the GPUs training at night

Sites (SITES below): the reference site (Llama-3.1-8B -> Llama-3.2-1B, ShareGPT request lengths)
and the second site (Llama-3.1-70B on four GPUs -> Mistral-24B) at the Azure trace's own chat
and code request lengths.

Appendix:
  programs_timescale  what grid programs require against the measured routing drop

Data: results/tables/datacenter/*.csv.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import P, SCI6, black, boolcol, read_dc, savefig

TZ = "America/New_York"          # the time axis of every time-series figure (PJM's)

# every flexible share 10-100 %, drawn as one cool -> warm gradient: 10 % flexible cool, 100 % warm
FLEX_SHARES = np.arange(1, 11) / 10
FLEX_CMAP = mpl.colors.LinearSegmentedColormap.from_list("flexibility", ["#2166ac", "#8860a8", "#b2182b"])
FLEX_NORM = mpl.colors.Normalize(0.1, 1.0)

# the single-column style of the three main figures (3.0 x 3.1 in, 7 pt, black text, y grid only)
RC_SINGLE_COLUMN = {
    "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.labelpad": 3, "axes.linewidth": 0.6,
    "text.color": "black", "axes.labelcolor": "black",
    "axes.edgecolor": "black", "axes.titlecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "legend.labelcolor": "black",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
}


@dataclass(frozen=True)
class Site:
    key: str                 # figure-name suffix
    tag: str                 # table-name suffix in results/tables/datacenter (datacenter_sim --only ...)
    label: str
    # per-figure drawing details that differ between the sites' figures
    floor_lw: float          # week: line width of each share's floor
    floor_alpha: float
    event_span_alpha: float  # event: the shaded 4-hour window
    event_line_alpha: float
    event_ylim: tuple | None  # None: (4, 1.1 x the week's peak)
    lend_fill_alpha: float   # GPUs leaving: the shaded training hours
    lend_ylim: tuple | None  # None: from the data
    fixed_day: str | None    # the busiest weekday; None: the trace's busiest day from the event table
    u_full_from_week: bool   # the always-full site's load from week.csv (the reference) or from the lending table


SITES = {
    "8b_1b_sharegpt": Site(
        key="8b_1b_sharegpt", tag="",
        label="reference site: 10,000 H100s, Llama-3.1-8B (7,500 GPUs) -> Llama-3.2-1B (2,500), ShareGPT request "
              "lengths, the Azure 2024 conversation trace's week of arrivals",
        floor_lw=0.45, floor_alpha=1.0, event_span_alpha=1.0, event_line_alpha=1.0, event_ylim=None,
        lend_fill_alpha=1.0, lend_ylim=(0, 2), fixed_day="2024-05-14", u_full_from_week=True),
    "70b_24b_azure_chat": Site(
        key="70b_24b_azure_chat", tag="_70b24b_chat",
        label="second site at Azure chat request lengths: Llama-3.1-70B (one instance over four GPUs; 1,875 instances) "
              "-> Mistral-24B (2,500 GPUs), mean 1,630 / 105 prompt / output tokens, the chat trace's week (12-18 May 2024)",
        floor_lw=0.15, floor_alpha=1.0, event_span_alpha=0.8, event_line_alpha=0.9, event_ylim=(5, 9),
        lend_fill_alpha=0.9, lend_ylim=None, fixed_day=None, u_full_from_week=False),
    "70b_24b_azure_code": Site(
        key="70b_24b_azure_code", tag="_70b24b_code",
        label="second site at Azure code request lengths: Llama-3.1-70B (one instance over four GPUs; 1,875 instances) "
              "-> Mistral-24B (2,500 GPUs), mean 2,464 / 22 prompt / output tokens, the code trace's week (10-16 May 2024)",
        floor_lw=0.45, floor_alpha=0.5, event_span_alpha=0.8, event_line_alpha=0.9, event_ylim=None,
        lend_fill_alpha=0.9, lend_ylim=None, fixed_day=None, u_full_from_week=False),
}


def _panel_style(a, offset_text=True):
    a.set_axisbelow(True)
    a.grid(False)
    a.grid(axis="y", color="#e5e5e5", lw=0.5)
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color("black")
    a.tick_params(length=2.5, width=0.6, colors="black")
    a.xaxis.label.set_color("black")
    a.yaxis.label.set_color("black")
    if offset_text:
        a.xaxis.get_offset_text().set_color("black")
        a.yaxis.get_offset_text().set_color("black")


def _share_colorbar(fig, cax_rect):
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=FLEX_NORM, cmap=FLEX_CMAP),
        cax=fig.add_axes(cax_rect),
        orientation="horizontal", ticks=[0.1, 1.0],
    )
    cb.ax.set_xticklabels(["10", "100"], color="black")
    cb.ax.set_title("Flexible queries (%)", fontsize=7, pad=3, color="black")
    cb.ax.tick_params(length=2, pad=1, labelsize=6.5, colors="black")
    cb.outline.set_visible(False)
    return cb


# ----------------------------------------------------------------------------------------------
def week_flexibility(site: str = "8b_1b_sharegpt"):
    """Top: facility power through the week in normal operation (black) and the floor routing
    reaches within seconds at every share of the queries flexible. Bottom left: a duration
    curve (at least y MW can be shed during x % of the week). Bottom right: the power that
    can be shed by time of day, averaged over the week."""
    S = SITES[site]
    wk = read_dc(f"week_by_share{S.tag}.csv", parse_dates=["t"])
    cmap, norm = FLEX_CMAP, FLEX_NORM
    base = wk[np.isclose(wk.flexible_share, 1)].sort_values("t").set_index("t")
    loc = base.index.tz_convert(TZ)

    with plt.rc_context(RC_SINGLE_COLUMN):
        fig = plt.figure(figsize=(3, 3.1))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1.15, 1], hspace=0.25, wspace=0.14,
            left=0.13, right=0.98, bottom=0.14, top=0.89,
        )
        ax = fig.add_subplot(gs[0, :])
        bx = fig.add_subplot(gs[1, 0])
        cx = fig.add_subplot(gs[1, 1], sharey=bx)
        ax.plot(loc, base.fac_base_mw, color="black", lw=1.1, zorder=5)
        ymax = 0

        for s in FLEX_SHARES:
            color = cmap(norm(s))
            w = wk[np.isclose(wk.flexible_share, s)].set_index("t").reindex(base.index)
            floor = np.minimum(
                w.fac_floor_mw.fillna(w.fac_base_mw).fillna(base.fac_base_mw),
                base.fac_base_mw,
            )
            ax.plot(loc, floor, color=color, lw=S.floor_lw, alpha=S.floor_alpha)

            v = w.fac_flex_mw.fillna(0).to_numpy()
            bx.plot(np.linspace(0, 100, len(v)), np.sort(v)[::-1], color=color, lw=1)
            ymax = max(ymax, v.max())

            lt = w.index.tz_convert(TZ)
            hour = np.round((lt.hour + lt.minute / 60) * 2) / 2
            daily = w.fac_flex_mw.groupby(hour).mean()
            cx.plot(daily.index, daily.values, color=color, lw=1)

        ax.set(ylabel="Facility power (MW)", xlim=(loc.min(), loc.max()))
        ax.margins(y=0.1)
        ax.xaxis.set_major_locator(mpl.dates.DayLocator(tz=TZ))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%a", tz=TZ))
        bx.set(
            xlabel="Availability (% of week)", ylabel="Power reduction (MW)",
            xlim=(0, 100), ylim=(0, ymax * 1.08), xticks=[0, 50, 100],
        )
        cx.set(xlabel="Hour of day", xlim=(0, 24), xticks=[0, 12, 24])
        cx.tick_params(axis="y", labelleft=False)

        for a in (ax, bx, cx):
            _panel_style(a)

        fig.legend(
            handles=[Line2D([], [], color="black", lw=1.1, label="Baseline")],
            loc="upper left", bbox_to_anchor=(0.12, 1.005),
            frameon=False, fontsize=7, handlelength=2, labelcolor="black",
        )
        _share_colorbar(fig, [0.55, 0.955, 0.41, 0.02])

        savefig(fig, f"datacenter_week_flexibility_{S.key}", "main")

    print(f"  {S.label}\n  top: weekly facility power; bottom left: duration curve; bottom right: mean daily flexibility")
    print(wk.groupby("flexible_share").agg(flex_mean_mw=("fac_flex_mw", lambda x: x.fillna(0).mean()),
                                           unserved=("fac_flex_mw", lambda x: x.isna().mean()),
                                           frac_mean=("flex_frac_fac", "mean")).round(3).to_string())


def event_day(site: str = "8b_1b_sharegpt"):
    """Top: the week's busiest day with a 16:00-20:00 (Eastern) event at every share of the
    queries flexible (normal operation in black). Bottom: the event's first and last minute at
    1 s resolution, the measured drop and return scaled to each share's cut."""
    S = SITES[site]
    ev = read_dc(f"event_by_share{S.tag}.csv", parse_dates=["t"])
    ev_sum = read_dc(f"event_summary_by_share{S.tag}.csv")
    step = read_dc(f"step_response{S.tag}.csv")
    step_sum = read_dc(f"step_summary{S.tag}.csv")
    cmap, norm = FLEX_CMAP, FLEX_NORM
    b = ev[np.isclose(ev.flexible_share, 1)].sort_values("t").reset_index(drop=True)
    loc = b.t.dt.tz_convert(TZ)
    day = loc.iloc[0].normalize()
    ticks = [day + pd.DateOffset(hours=h) for h in (12, 16, 20, 24)]
    on = b[boolcol(b, "in_event")]
    t0, t1 = on.t.min(), on.t.max() + pd.Timedelta("5min")
    shape = step[step.workload == "sharegpt"].set_index("t_rel_s").y_p50.sort_index()
    ss = step_sum.set_index("workload").loc["sharegpt"]

    with plt.rc_context(RC_SINGLE_COLUMN):
        fig = plt.figure(figsize=(3, 3.1))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1.15, 1], hspace=0.25, wspace=0.14,
            left=0.13, right=0.98, bottom=0.14, top=0.89,
        )
        ax = fig.add_subplot(gs[0, :])
        bx = fig.add_subplot(gs[1, 0])
        cx = fig.add_subplot(gs[1, 1], sharey=bx)

        ax.axvspan(t0.tz_convert(TZ), t1.tz_convert(TZ),
                   color="#e5e5e5", alpha=S.event_span_alpha, lw=0, zorder=0)
        ax.plot(loc, b.fac_base_mw, color="black", lw=1.1, zorder=5)

        for s in FLEX_SHARES:
            color = cmap(norm(s))
            d = ev[np.isclose(ev.flexible_share, s)].sort_values("t")
            near = d.t.between(t0 - pd.Timedelta("15min"), t1 + pd.Timedelta("15min"))
            ax.plot(d.t[near].dt.tz_convert(TZ), d.fac_mw[near],
                    color=color, lw=0.85, alpha=S.event_line_alpha)

            o = d[boolcol(d, "in_event")]
            for a, lo, hi, off, idx in (
                (bx, -10, 45, 0, 0),
                (cx, 40, 100, 50, -1),
            ):
                p0, p1 = float(o.fac_base_mw.iloc[idx]), float(o.fac_mw.iloc[idx])
                y = shape.loc[(shape.index >= lo) & (shape.index <= hi)]
                a.plot(y.index - off, p0 + y.to_numpy() * (p1 - p0),
                       color=color, lw=1.2)

        ax.set(ylabel="Facility power (MW)", ylim=S.event_ylim or (4, b.fac_base_mw.max() * 1.1))
        ax.set_xticks(ticks, ["12:00", "16:00", "20:00", "24:00"])
        ax.set_xlim(ticks[0], ticks[-1])
        bx.set(xlabel="Time from start (s)", ylabel="Facility power (MW)", xlim=(-10, 45))
        cx.set(xlabel="Time from end (s)", xlim=(-10, 50))
        cx.tick_params(axis="y", labelleft=False)

        for a in (bx, cx):
            a.axvline(0, color="black", lw=1, ls=":", alpha=0.6)

        for a in (ax, bx, cx):
            _panel_style(a)

        fig.legend(
            handles=[Line2D([], [], color="black", lw=1.1, label="Baseline")],
            loc="upper left", bbox_to_anchor=(0.12, 1.005),
            frameon=False, fontsize=7, handlelength=2, labelcolor="black",
        )
        _share_colorbar(fig, [0.55, 0.955, 0.41, 0.02])

        savefig(fig, f"datacenter_event_day_{S.key}", "main")

    projected = "" if S.tag == "" else f" (projected: measured shape stretched {ss.stretch:.2f}x)"
    print(f"  {S.label}\n  top: four-hour event on {loc.iloc[0]:%a %d %b %Y}, 12:00-24:00 (US Eastern)")
    print(f"  bottom left: event start; 90 % reduction in {ss.t_down90_s:.0f} s{projected}")
    print(f"  bottom right: event end; 90 % recovery in {ss.t_back90_s:.0f} s")
    cols = ["flexible_share"] + (["depth"] if "depth" in ev_sum else []) + \
        ["fac_shed_mean_mw", "shed_frac_mean", "shed_mwh", "queries_moved_frac_of_event", "u_event_mean"]
    print(ev_sum[cols].round(3).to_string(index=False))


def gpus_leaving(site: str = "8b_1b_sharegpt"):
    """Top left: week-mean power reduction against the inference GPUs still available (traffic
    unchanged, both pools alike), every flexible share; dotted from the first point at which
    some busy 5-minute steps are no longer served. Top right: the load on the inference GPUs
    through the busiest weekday, with every GPU and with 40 % of them training at night
    (shaded). Bottom: the power reduction through the same day, with every GPU (faint) and with
    the 40 % lent (solid), at half and all of the queries flexible."""
    S = SITES[site]
    cmap, norm = FLEX_CMAP, FLEX_NORM
    avs = read_dc(f"availability_by_share{S.tag}.csv")
    lend = read_dc(f"week_lending{S.tag}.csv", parse_dates=["t"])
    week = read_dc("week.csv", parse_dates=["t"]) if S.u_full_from_week else None
    desc = lambda d: d.sort_values("available_frac", ascending=False)  # noqa: E731
    DOWN = "#fde0c2"                     # light orange: the hours with 40 % of the GPUs down (60 % available)
    if S.fixed_day:
        day0 = pd.Timestamp(S.fixed_day, tz=TZ)
    else:                                # the trace's busiest day (the event figure's)
        day0 = pd.Timestamp(read_dc(f"event_summary_by_share{S.tag}.csv").day.iloc[0], tz=TZ)

    def one_day(s):
        g = lend[np.isclose(lend.flexible_share, s)]
        lt = g.t.dt.tz_convert(TZ)
        g = g[(lt >= day0) & (lt < day0 + pd.Timedelta("1D"))].copy()
        lt = g.t.dt.tz_convert(TZ)
        g["hour"] = (lt.dt.hour + lt.dt.minute / 60).to_numpy()
        if S.u_full_from_week:
            g["u_full"] = g.t.map(week.set_index("t").u).to_numpy()
        g["lent"] = (g.inference_gpus < g.inference_gpus.max()).to_numpy()
        return g

    with plt.rc_context(RC_SINGLE_COLUMN):
        fig = plt.figure(figsize=(3, 3.1))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1, 1.15], hspace=0.55, wspace=0.5,
            left=0.13, right=0.98, bottom=0.12, top=0.89,
        )
        ax = fig.add_subplot(gs[0, 0])
        ux = fig.add_subplot(gs[0, 1])
        bx = fig.add_subplot(gs[1, :])
        days = {s: one_day(s) for s in (0.5, 1.0)}
        lent = days[0.5].lent.to_numpy()
        hour = days[0.5].hour.to_numpy()

        # top left: week-mean reduction against the GPUs still available, every flexible share
        for s in FLEX_SHARES:
            g = desc(avs[np.isclose(avs.flexible_share, s)])
            x, y = 100 * g.available_frac.to_numpy(), g.fac_flex_mean_mw.to_numpy()
            ok = g.served_frac_of_week.to_numpy() >= 0.999
            k = int(np.argmin(ok)) if (~ok).any() else len(ok)
            ax.plot(x[:k], y[:k], color=cmap(norm(s)), lw=0.9)
            ax.plot(x[max(k - 1, 0):], y[max(k - 1, 0):], color=cmap(norm(s)), lw=0.7, ls=(0, (1, 1.2)))
        ax.set(xlabel="GPUs available (%)", ylabel="Power reduction (MW)",
               xlim=(100, 50), ylim=(0, None), xticks=[100, 75, 50])

        # top right: the load on the inference GPUs -- the same traffic on 60 % of them during the shaded hours
        for a in (ux, bx):
            a.fill_between(hour, 0, 1, where=lent, step="post", transform=a.get_xaxis_transform(),
                           color=DOWN, alpha=S.lend_fill_alpha, lw=0, zorder=0)
        ux.plot(hour, 100 * days[0.5].u_full, color="#888888", lw=0.8, alpha=0.6)
        ux.plot(hour, 100 * days[0.5].u, color="black", lw=0.9)
        ux.set(xlabel="Hour of day", ylabel="Load (% of capacity)",
               xlim=(0, 24), ylim=(0, 100), xticks=[0, 12, 24], yticks=[0, 50, 100])

        # bottom: the power reduction through the same day, half and all of the queries flexible
        for s in (0.5, 1.0):
            d = days[s]
            bx.plot(d.hour, d.fac_flex_mw_always_full, color=cmap(norm(s)), lw=0.8, alpha=0.5)
            bx.plot(d.hour, d.fac_flex_mw, color=cmap(norm(s)), lw=1.0)
        if S.lend_ylim is not None:
            bx.set(ylabel="Power reduction (MW)", xlim=(0, 24), ylim=S.lend_ylim)
        else:
            top = max(d.fac_flex_mw_always_full.max() for d in days.values())
            bx.set(ylabel="Power reduction (MW)", xlim=(0, 24), ylim=(0, np.ceil(top * 1.08 * 10) / 10))
        bx.set_xticks([0, 6, 12, 18, 24], ["0:00", "6:00", "12:00", "18:00", "24:00"])
        bx.set_xlabel("Hour of day", labelpad=1.5)

        for a in (ax, ux, bx):
            _panel_style(a, offset_text=False)

        fig.legend(
            handles=[Line2D([], [], color="#888888", lw=0.8, alpha=0.6),
                     (Patch(facecolor=DOWN, edgecolor="none"), Line2D([], [], color="black", lw=1.0))],
            labels=["All GPUs", "60% GPUs available"],
            handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},     # the line drawn over the shaded box
            loc="upper left", bbox_to_anchor=(0.12, 1.005), ncol=1,
            frameon=False, fontsize=6.5, handlelength=1.8, labelcolor="black", labelspacing=0.2,
        )
        _share_colorbar(fig, [0.62, 0.955, 0.34, 0.02])

        savefig(fig, f"datacenter_gpus_leaving_{S.key}", "main")

    n = days[0.5][days[0.5].lent]
    print(f"  {S.label}\n  top left: week-mean power reduction against inference GPUs available (traffic unchanged, both pools alike)")
    print(f"  top right: load on the inference GPUs, {day0:%a %d %b %Y}; shaded hours {100 * n.u_full.min():.0f}-{100 * n.u_full.max():.0f} % "
          f"with every GPU -> {100 * n.u.min():.0f}-{100 * n.u.max():.0f} % with 40 % training")
    print("  bottom: power reduction through the same day; half (purple) and all (red) of the queries flexible")


# ---- appendix ---------------------------------------------------------------------------------
def programs_timescale():
    """What grid products and large-load agreements require -- time to full response (marker)
    and how long the response is held (grey bar) -- against the measured 90 % drop time of
    routing (shaded band, by task)."""
    programs = read_dc("grid_programs.csv")
    step_sum = read_dc("step_summary.csv")
    SHORT = {"ercot_ffr": "ERCOT Fast Frequency Response", "ercot_rrs_ufr": "ERCOT RRS, under-frequency relay",
             "ercot_regulation": "ERCOT Regulation", "pjm_regulation": "PJM Regulation",
             "ercot_ecrs": "ERCOT Contingency Reserve", "pjm_sync_reserve": "PJM Synchronized Reserve",
             "ercot_ers10": "ERCOT Emergency Response (ERS-10)", "ercot_nonspin": "ERCOT Non-Spinning Reserve",
             "pjm_load_mgmt": "PJM Load Management", "sb6_netmetered": "Texas SB 6 large-load curtailment",
             "tva_xai": "TVA–xAI approval condition", "emerald_phoenix": "Field demo: Phoenix (Emerald AI)",
             "emerald_london": "Field demo: London (Emerald AI)"}
    MEET, MISS = SCI6[2], SCI6[3]
    pr = programs.sort_values(["response_s", "duration_s"], ascending=[False, False]).reset_index(drop=True)
    lo, hi = step_sum.t_down90_s.min(), step_sum.t_down90_s.max()
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.axvspan(lo, hi, color=SCI6[0], alpha=0.18, lw=0, zorder=0)
    for i, r in pr.iterrows():
        ok = bool(r.routing_meets_response)
        if np.isfinite(r.duration_s):
            ax.plot([r.response_s, r.duration_s], [i, i], color="#bdbdbd", lw=4.0, solid_capstyle="butt", zorder=1)
        else:                                            # held until the grid releases it: no limit stated
            ax.plot([r.response_s, 2e5], [i, i], color="#bdbdbd", lw=4.0, ls=(0, (0.5, 0.5)), zorder=1)
        ax.scatter([r.response_s], [i], s=34, marker="o" if ok else "X", color=MEET if ok else MISS, zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(pr)), [SHORT.get(p, l) for p, l in zip(pr["product"], pr.label)], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(0.1, 2e5); ax.set_ylim(-0.6, len(pr) - 0.4)
    ax.set_xticks([0.1, 1, 10, 60, 600, 3600, 36000], ["0.1 s", "1 s", "10 s", "1 min", "10 min", "1 h", "10 h"])
    ax.minorticks_off()
    ax.set_xlabel("Time (log scale)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
    black(ax, title=9, label=9, tick=7.5)
    n_ok = int(pr.routing_meets_response.astype(bool).sum())
    handles = [Patch(facecolor=SCI6[0], alpha=0.18, edgecolor="none", label="Response Time via Routing (in seconds)"),
               Line2D([], [], marker="o", color=MEET, ls="none", ms=6, label=f"Required response, routing sufficient ({n_ok})"),
               Line2D([], [], marker="X", color=MISS, ls="none", ms=6, label=f"Required response, routing insufficient ({len(pr) - n_ok})"),
               Line2D([], [], color="#bdbdbd", lw=4, label="Duration for Curtailment"),
               Line2D([], [], color="#bdbdbd", lw=4, ls=(0, (0.5, 0.5)), label="Until released (no limit stated)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=2, frameon=False, fontsize=7.5, labelcolor="black")
    fig.subplots_adjust(left=0.36, right=0.98, top=0.98, bottom=0.27)
    savefig(fig, "grid_programs_response_timescale", "appendix")
    print(f"  routing's measured 90 % drop time {lo:.0f}-{hi:.0f} s by task; fast enough for {n_ok} of {len(pr)} programs")


def main_figures():
    for site in SITES:
        week_flexibility(site)
    for site in SITES:
        event_day(site)
    for site in SITES:
        gpus_leaving(site)


MAIN = [main_figures]
APPENDIX = [programs_timescale]
