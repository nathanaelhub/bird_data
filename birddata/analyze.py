"""
Analysis + figures for the bird-collision study.

Reads the processed files from etl.py and answers the questions the story
turns on, writing one figure each to figures/:

    seasonal.png         when do collisions happen?        (migration timing)
    light_mortality.png  do brighter nights kill more?      (the conservation lever)
    species.png          which birds die?                   (who is vulnerable)
    concentration.png    how concentrated is the toll?      (where to intervene)
    trend.png            is per-night mortality changing?    (long-term direction)

Run `bird-etl` first.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # repo root (package lives in birddata/)
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

# palette (validated for a white surface; sequential ramp is light -> dark)
INK = "#334155"
MUTED = "#94a3b8"
GRID = "#e2e8f0"
ACCENT = "#0d9488"
RAMP = ["#99e6d9", "#2dbfa8", "#0d7d6e"]
MONTHS = {3: "Mar", 4: "Apr", 5: "May", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}


def _style(ax, title, subtitle=None):
    # header lives in figure space above the reserved plot area (see tight_layout
    # rect in each figure), so the title and subtitle never crowd the axes
    fig = ax.get_figure()
    fig.text(0.035, 0.96, title, fontsize=14, fontweight="bold", color=INK, va="top", ha="left")
    if subtitle:
        fig.text(0.035, 0.885, subtitle, fontsize=10, color=MUTED, va="top", ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.set_axisbelow(True)


def correlation_stats(x, y):
    """Pearson r with a two-sided p-value and 95% CI — no scipy dependency.

    Pairs with NaNs in either series are dropped first. The confidence interval
    and the p-value both come from Fisher's z-transform; the p-value is the
    normal approximation, which here (n in the thousands) agrees with the exact
    t-test to many decimals. Returns a dict with keys ``r``, ``n``, ``p``,
    ``ci_low``, ``ci_high`` — all NaN when there are too few points or either
    series is constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(len(x))
    nan = float("nan")
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": nan, "n": n, "p": nan, "ci_low": nan, "ci_high": nan}
    r = float(np.corrcoef(x, y)[0, 1])
    rc = min(max(r, -0.999999), 0.999999)  # guard atanh at the poles
    z = math.atanh(rc)
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    # two-sided p via the normal CDF: 2*(1 - Phi(|z|/se)) == erfc(|z|/se / sqrt2)
    p = math.erfc(abs(z) / se / math.sqrt(2))
    return {"r": r, "n": n, "p": float(p), "ci_low": float(lo), "ci_high": float(hi)}


def _fmt_p(p):
    """Compact p-value label for a figure annotation or printed headline."""
    if p != p:  # NaN
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fig_seasonal(coll):
    by_month = coll["month"].value_counts().reindex(MONTHS).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    bars = ax.bar(
        [MONTHS[m] for m in by_month.index], by_month.values, color=ACCENT, width=0.72, zorder=3
    )
    ax.bar_label(
        bars, labels=[f"{int(v):,}" for v in by_month.values], padding=3, fontsize=9.5, color=INK
    )
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_ylim(0, by_month.max() * 1.15)
    ax.set_yticks([])
    _style(
        ax,
        "Collisions concentrate in the migration windows",
        "Chicago bird–building collisions by month, 1978–2016",
    )
    fig.text(
        0.5,
        0.02,
        "Spring (Mar–May) and fall (Aug–Nov) passage account for nearly every recorded death.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "seasonal.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_light(daily):
    st = correlation_stats(daily["light_score"], daily["collisions"])
    daily = daily.copy()
    daily["tercile"] = pd.qcut(daily["light_score"], 3, labels=["Low", "Medium", "High"])
    means = daily.groupby("tercile", observed=True)["collisions"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    bars = ax.bar(means.index.astype(str), means.values, color=RAMP, width=0.66, zorder=3)
    ax.bar_label(
        bars,
        labels=[f"{v:.1f}" for v in means.values],
        padding=3,
        fontsize=10.5,
        color=INK,
        fontweight="bold",
    )
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_ylim(0, means.max() * 1.18)
    ax.set_yticks([])
    ax.set_xlabel("Building light level that night", fontsize=10, color=MUTED, labelpad=8)
    _style(
        ax,
        "Brighter nights kill more birds",
        "Mean collisions per night at McCormick Place, by light level",
    )
    # stats sit top-left, over the short "Low" bar, clear of the taller bars' labels
    ax.text(
        0.02,
        0.95,
        f"Pearson r = {st['r']:.2f}",
        transform=ax.transAxes,
        ha="left",
        fontsize=10,
        color=ACCENT,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.87,
        f"95% CI {st['ci_low']:.2f}–{st['ci_high']:.2f} · {_fmt_p(st['p'])} (n={st['n']:,})",
        transform=ax.transAxes,
        ha="left",
        fontsize=8.5,
        color=MUTED,
    )
    fig.text(
        0.5,
        0.02,
        "Nights in the brightest third average roughly twice the deaths of the darkest third.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "light_mortality.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_species(coll):
    top = coll["family"].value_counts().head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    bars = ax.barh(top.index, top.values, color=ACCENT, height=0.7, zorder=3)
    ax.bar_label(
        bars, labels=[f"{int(v):,}" for v in top.values], padding=4, fontsize=9.5, color=INK
    )
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0, top.max() * 1.13)
    ax.set_xticks([])
    _style(
        ax, "A few migratory families take most of the toll", "Collisions by bird family (top 8)"
    )
    share = (coll["flight_call"] == "Yes").mean()
    fig.text(
        0.5,
        0.02,
        f"{share:.0%} of all collisions are nocturnal "
        "flight-calling species — the birds drawn to lit windows.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "species.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mp_nights(coll):
    """McCormick Place collision counts per night, deadliest first."""
    return (
        coll[coll["locality"] == "McCormick Place"]
        .groupby("date")
        .size()
        .sort_values(ascending=False)
        .values
    )


def concentration_share(nights, frac):
    """Share of all deaths falling on the deadliest ``frac`` of nights.

    ``nights`` is a sequence of per-night counts (any order); ``frac`` is in
    (0, 1]. Returns the cumulative fraction of deaths on the worst
    ``max(1, floor(frac * n))`` nights — e.g. ``frac=0.10`` answers "what share
    of deaths happen on the worst 10% of nights?". Underpins the concentration
    figure and the printed headline so both report the same number.
    """
    nights = np.sort(np.asarray(nights, dtype=float))[::-1]
    total = nights.sum()
    if len(nights) == 0 or total == 0:
        return 0.0
    k = max(1, int(len(nights) * frac))
    return float(nights[:k].sum() / total)


def fig_concentration(coll):
    nights = np.sort(_mp_nights(coll))[::-1]
    total = nights.sum()
    cum = np.concatenate([[0.0], np.cumsum(nights) / total])
    x = np.arange(len(nights) + 1) / len(nights)
    y10 = concentration_share(nights, 0.10)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    # reference line: what the curve would look like if deaths were spread evenly
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.fill_between(x, cum, color=ACCENT, alpha=0.08, zorder=1)
    ax.plot(x, cum, color=ACCENT, linewidth=2.2, zorder=4)
    ax.plot([0.10, 0.10], [0, y10], color=INK, linestyle=":", linewidth=1, zorder=3)
    ax.scatter([0.10], [y10], s=44, color=ACCENT, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"worst 10% of nights\ncause {y10:.0%} of deaths",
        xy=(0.10, y10),
        xytext=(0.24, y10 - 0.15),
        fontsize=10.5,
        color=INK,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{int(t * 100)}%" for t in ticks])
    ax.set_yticklabels([f"{int(t * 100)}%" for t in ticks])
    ax.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_xlabel(
        "share of monitored nights (deadliest first)", fontsize=10, color=MUTED, labelpad=8
    )
    ax.set_ylabel("cumulative share of deaths", fontsize=10, color=MUTED, labelpad=8)
    _style(
        ax,
        "The toll is concentrated on a few catastrophic nights",
        "Cumulative deaths vs. nights at McCormick Place (dashed = deaths spread evenly)",
    )
    fig.text(
        0.5,
        0.02,
        "Half the birds die on a tenth of the nights — so targeting the "
        "highest-risk nights captures most of the benefit.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "concentration.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_trend(daily):
    d = daily.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    by_year = d.groupby("year")["collisions"].mean()
    years = by_year.index.values.astype(float)
    means = by_year.values
    # effort is controlled by using per-night rates; significance of the drift is
    # tested on every monitored night, not the 17 yearly means
    st = correlation_stats(d["year"], d["collisions"])
    slope = float(np.polyfit(years, means, 1)[0])
    fit = np.poly1d(np.polyfit(years, means, 1))

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    ax.plot(years, fit(years), color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.plot(
        years,
        means,
        color=ACCENT,
        linewidth=2,
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=1,
        zorder=4,
    )
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_ylim(0, means.max() * 1.2)
    ax.set_xticks(years[::2])
    ax.set_xticklabels([f"{int(y)}" for y in years[::2]])
    ax.set_ylabel("mean collisions per monitored night", fontsize=10, color=MUTED, labelpad=8)
    _style(
        ax,
        "Per-night mortality drifts down slightly over the record",
        "McCormick Place deaths per monitored night, by year",
    )
    ax.text(
        0.98,
        0.95,
        f"slope {slope:+.2f}/yr · {_fmt_p(st['p'])}",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color=ACCENT,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "A modest, statistically real decline — but night-to-night "
        "variation dwarfs it, so year explains under 1% of the spread.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "trend.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if not (PROC / "collisions_clean.csv").exists():
        sys.exit("processed data missing — run `bird-etl` first.")
    coll = pd.read_csv(PROC / "collisions_clean.csv", parse_dates=["date"])
    daily = pd.read_csv(PROC / "mp_daily.csv", parse_dates=["date"])
    FIG.mkdir(exist_ok=True)

    fig_seasonal(coll)
    fig_light(daily)
    fig_species(coll)
    fig_concentration(coll)
    fig_trend(daily)

    # headline findings, printed for the record
    light = correlation_stats(daily["light_score"], daily["collisions"])
    # Spearman as a rank-Pearson, so we don't pull in scipy for one number
    rs = daily["light_score"].rank().corr(daily["collisions"].rank())
    trend = correlation_stats(pd.to_datetime(daily["date"]).dt.year, daily["collisions"])
    years = pd.to_datetime(daily["date"]).dt.year
    share = (coll["flight_call"] == "Yes").mean()
    worst10 = concentration_share(_mp_nights(coll), 0.10)
    print(f"records analysed : {len(coll):,} collisions, {len(daily):,} MP nights")
    print(
        f"light vs deaths  : Pearson r={light['r']:.3f} "
        f"(95% CI {light['ci_low']:.2f}–{light['ci_high']:.2f}, {_fmt_p(light['p'])}), "
        f"Spearman={rs:.3f}"
    )
    print(f"flight-callers   : {share:.1%} of all collisions")
    print(
        f"peak month       : {coll['month'].value_counts().idxmax()} "
        f"({MONTHS[coll['month'].value_counts().idxmax()]})"
    )
    print(f"concentration    : worst 10% of nights = {worst10:.0%} of all deaths")
    print(
        f"per-night trend  : r(year,deaths)={trend['r']:+.3f} ({_fmt_p(trend['p'])}), "
        f"{years.min()}–{years.max()}"
    )
    print(f"figures written  : {', '.join(p.name for p in sorted(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
