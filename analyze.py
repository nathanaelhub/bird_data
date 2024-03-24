"""
Analysis + figures for the bird-collision study.

Reads the processed files from etl.py and answers the three questions the story
turns on, writing one figure each to figures/:

    seasonal.png         when do collisions happen?        (migration timing)
    light_mortality.png  do brighter nights kill more?      (the conservation lever)
    species.png          which birds die?                   (who is vulnerable)

Run `python etl.py` first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
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
    fig.text(0.035, 0.96, title, fontsize=14, fontweight="bold", color=INK,
             va="top", ha="left")
    if subtitle:
        fig.text(0.035, 0.885, subtitle, fontsize=10, color=MUTED,
                 va="top", ha="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.set_axisbelow(True)


def fig_seasonal(coll):
    by_month = coll["month"].value_counts().reindex(MONTHS).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    bars = ax.bar([MONTHS[m] for m in by_month.index], by_month.values,
                  color=ACCENT, width=0.72, zorder=3)
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in by_month.values],
                 padding=3, fontsize=9.5, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_ylim(0, by_month.max() * 1.15)
    ax.set_yticks([])
    _style(ax, "Collisions concentrate in the migration windows",
           "Chicago bird–building collisions by month, 1978–2016")
    fig.text(0.5, 0.02, "Spring (Mar–May) and fall (Aug–Nov) passage account for "
             "nearly every recorded death.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "seasonal.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_light(daily):
    r = daily["light_score"].corr(daily["collisions"])
    daily = daily.copy()
    daily["tercile"] = pd.qcut(daily["light_score"], 3,
                               labels=["Low", "Medium", "High"])
    means = daily.groupby("tercile", observed=True)["collisions"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    bars = ax.bar(means.index.astype(str), means.values, color=RAMP,
                  width=0.66, zorder=3)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in means.values],
                 padding=3, fontsize=10.5, color=INK, fontweight="bold")
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_ylim(0, means.max() * 1.18)
    ax.set_yticks([])
    ax.set_xlabel("Building light level that night", fontsize=10, color=MUTED, labelpad=8)
    _style(ax, "Brighter nights kill more birds",
           "Mean collisions per night at McCormick Place, by light level")
    ax.text(0.98, 0.94, f"Pearson r = {r:.2f}", transform=ax.transAxes,
            ha="right", fontsize=10, color=ACCENT, fontweight="bold")
    fig.text(0.5, 0.02, "Nights in the brightest third average roughly twice the "
             "deaths of the darkest third.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "light_mortality.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_species(coll):
    top = coll["family"].value_counts().head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    bars = ax.barh(top.index, top.values, color=ACCENT, height=0.7, zorder=3)
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in top.values],
                 padding=4, fontsize=9.5, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0, top.max() * 1.13)
    ax.set_xticks([])
    _style(ax, "A few migratory families take most of the toll",
           "Collisions by bird family (top 8)")
    share = (coll["flight_call"] == "Yes").mean()
    fig.text(0.5, 0.02, f"{share:.0%} of all collisions are nocturnal "
             "flight-calling species — the birds drawn to lit windows.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.savefig(FIG / "species.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if not (PROC / "collisions_clean.csv").exists():
        sys.exit("processed data missing — run `python etl.py` first.")
    coll = pd.read_csv(PROC / "collisions_clean.csv", parse_dates=["date"])
    daily = pd.read_csv(PROC / "mp_daily.csv", parse_dates=["date"])
    FIG.mkdir(exist_ok=True)

    fig_seasonal(coll)
    fig_light(daily)
    fig_species(coll)

    # headline findings, printed for the record
    r = daily["light_score"].corr(daily["collisions"])
    # Spearman as a rank-Pearson, so we don't pull in scipy for one number
    rs = daily["light_score"].rank().corr(daily["collisions"].rank())
    share = (coll["flight_call"] == "Yes").mean()
    print(f"records analysed : {len(coll):,} collisions, {len(daily):,} MP nights")
    print(f"light vs deaths  : Pearson r={r:.3f}, Spearman={rs:.3f}")
    print(f"flight-callers   : {share:.1%} of all collisions")
    print(f"peak month       : {coll['month'].value_counts().idxmax()} "
          f"({MONTHS[coll['month'].value_counts().idxmax()]})")
    print(f"figures written  : {', '.join(p.name for p in sorted(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
