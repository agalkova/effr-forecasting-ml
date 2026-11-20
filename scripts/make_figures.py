"""Draw the figures for the external test period.

Reads the committed results in reports/external/ and writes SVG figures to
docs/figures/. Each figure is drawn at the width it is shown at: a wide version and a narrow one for
small screens.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs
from figure_style import plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "external"
OUT = ROOT / "docs" / "figures"

PERIODS = [
    ("Mar 2017 - Dec 2021", "2017-03", "2021-12"),
    ("2022-23 rate hikes", "2022-01", "2023-12"),
    ("Jan 2024 - Aug 2026", "2024-01", "2026-12"),
]
MODELS = [
    ("Persistence", "Repeat last month", fs.GREY),
    ("Ridge", "Ridge", fs.BLUE),
    ("RandomForest", "Random Forest", fs.ORANGE),
]


def load() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def period_rmse(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end in PERIODS:
        part = frame[
            (frame["date"] >= start) & (frame["date"] <= pd.Timestamp(end) + pd.offsets.MonthEnd(0))
        ]
        for column, name, _ in MODELS:
            rows.append(
                {
                    "period": label,
                    "model": name,
                    "months": len(part),
                    "rmse": rmse(part[column] - part["Actual"]),
                }
            )
    return pd.DataFrame(rows)


def forecast_figure(frame: pd.DataFrame, narrow: bool) -> None:
    fig = fs.new_figure(6.4, 7.2)
    top = fig.add_axes(
        [
            0.13 if narrow else 0.075,
            0.43 if narrow else 0.42,
            0.83 if narrow else 0.9,
            0.37 if narrow else 0.33,
        ]
    )
    bottom = fig.add_axes(
        [0.13 if narrow else 0.075, 0.09, 0.83 if narrow else 0.9, 0.24 if narrow else 0.25],
        sharex=top,
    )
    dates = frame["date"]
    hike = (pd.Timestamp("2022-03-01"), pd.Timestamp("2023-07-31"))
    for axis in (top, bottom):
        axis.axvspan(*hike, color=fs.SHADE, zorder=0, lw=0)

    top.plot(dates, frame["Actual"], color="#c5cdd3", lw=6, label="Actual rate", zorder=1)
    top.plot(dates, frame["RandomForest"], color=fs.ORANGE, lw=1.4, label="Random Forest", zorder=2)
    top.plot(dates, frame["Ridge"], color=fs.BLUE, lw=1.6, label="Ridge", zorder=3)
    top.set_ylim(-0.2, 6)
    top.set_yticks(range(0, 7, 2 if narrow else 1))
    top.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    top.tick_params(labelbottom=False)
    if narrow:
        top.legend(
            loc="lower left",
            ncol=3,
            bbox_to_anchor=(-0.02, 1.0),
            columnspacing=0.9,
            handlelength=1.2,
            fontsize=plt.rcParams["font.size"] - 2,
        )
    else:
        top.legend(loc="lower left", ncol=3, bbox_to_anchor=(-0.01, 1.0))

    bottom.axhline(0, color=fs.SPINE, lw=0.8)
    bottom.plot(
        dates,
        frame["Persistence"] - frame["Actual"],
        color=fs.GREY,
        lw=1.3,
        label="Repeat last month",
    )
    bottom.plot(dates, frame["Ridge"] - frame["Actual"], color=fs.BLUE, lw=1.5, label="Ridge")
    bottom.set_ylim(-1.15, 1.15)
    bottom.set_yticks([-1, 0, 1] if narrow else [-1, -0.5, 0, 0.5, 1])
    bottom.yaxis.set_major_formatter(lambda v, _: f"{v:+.1f}" if v else "0")
    if narrow:
        bottom.legend(
            loc="lower left",
            ncol=2,
            bbox_to_anchor=(-0.02, 1.0),
            columnspacing=0.9,
            handlelength=1.2,
            fontsize=plt.rcParams["font.size"] - 2,
        )
        bottom.text(
            -0.02,
            1.2,
            "Miss each month (forecast - actual)",
            transform=bottom.transAxes,
            color=fs.INK,
            fontweight="bold",
            fontsize=plt.rcParams["font.size"] - 1,
        )
    else:
        bottom.legend(
            loc="lower right",
            ncol=2,
            bbox_to_anchor=(1.0, 1.0),
            fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        )
        bottom.text(
            0.0,
            1.06,
            "Miss each month (forecast - actual)",
            transform=bottom.transAxes,
            color=fs.INK,
            fontweight="bold",
            fontsize=plt.rcParams["font.size"],
        )
    covid = frame.loc[(frame["Ridge"] - frame["Actual"]).idxmax()]
    if not narrow:
        bottom.annotate(
            "Mar 2020 cut",
            xy=(covid["date"], covid["Ridge"] - covid["Actual"]),
            xytext=(16, -6),
            textcoords="offset points",
            color=fs.MUTED,
            fontsize=plt.rcParams["font.size"] - 1.5,
            va="center",
            arrowprops={"arrowstyle": "-", "color": fs.SPINE, "lw": 0.8},
        )
    if narrow:
        import matplotlib.dates as mdates

        bottom.xaxis.set_major_locator(mdates.YearLocator(2))
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))

    fs.header(
        fig,
        1,
        "Actual rate and one-month-ahead forecasts"
        if not narrow
        else "Forecasts vs the actual rate",
        "Mar 2017 - Aug 2026, 114 months"
        if narrow
        else "External test, Mar 2017 - Aug 2026 (114 months). Shaded: 2022-23 rate hikes",
        narrow,
    )
    fs.source(fig, "reports/external/predictions.csv", narrow)
    name = "external-forecasts" + fs.suffix()
    fs.save(
        fig,
        OUT / f"{name}.svg",
        "Actual federal funds rate and one-month-ahead forecasts",
        "Top: actual rate with Ridge and Random Forest forecasts, March 2017 to August 2026. "
        "Bottom: each month's miss for Ridge and for repeating last month's rate. Misses stay near zero "
        "except around the March 2020 cut and during the 2022-23 rate increases.",
    )


def period_figure(frame: pd.DataFrame, narrow: bool) -> None:
    table = period_rmse(frame)
    fig = fs.new_figure(4.1, 5.2)
    axis = fig.add_axes(
        [
            0.15 if narrow else 0.075,
            0.17 if narrow else 0.14,
            0.81 if narrow else 0.9,
            0.54 if narrow else 0.6,
        ]
    )
    n_models = len(MODELS)
    bar = 0.26
    for k, (_, name, color) in enumerate(MODELS):
        values = table[table["model"] == name]["rmse"].to_numpy()
        xs = np.arange(len(PERIODS)) + (k - (n_models - 1) / 2) * (bar + 0.02)
        axis.bar(xs, values, width=bar, color=color, label=name, zorder=2)
        for x, v in zip(xs, values):
            axis.text(
                x,
                v + 0.006,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                color=fs.INK_SOFT,
                fontsize=plt.rcParams["font.size"] - (2.5 if narrow else 1.5),
            )
    months = table.groupby("period", sort=False)["months"].first().to_numpy()
    labels = [f"{p[0]}\n{m} months" for p, m in zip(PERIODS, months)]
    if narrow:
        labels = ["2017-21\n58 mo", "2022-23\n24 mo", "2024-26\n32 mo"]
    axis.set_xticks(range(len(PERIODS)), labels)
    axis.tick_params(axis="x", length=0)
    axis.set_ylim(0, 0.42)
    axis.set_ylabel("RMSE (pp)")
    if narrow:
        axis.legend(
            loc="lower left",
            ncol=2,
            bbox_to_anchor=(-0.02, 1.0),
            columnspacing=0.9,
            handlelength=1.2,
            fontsize=plt.rcParams["font.size"] - 2,
        )
    else:
        axis.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.04))
    fs.header(
        fig,
        2,
        "Forecast error by period"
        if narrow
        else "Forecast error by period (RMSE, lower is better)",
        None if narrow else "Same 114 test months, split into three stretches",
        narrow,
    )
    fs.source(fig, "reports/external/predictions.csv", narrow)
    name = "error-by-period" + fs.suffix()
    fs.save(
        fig,
        OUT / f"{name}.svg",
        "Forecast error by period",
        "RMSE in percentage points for repeating last month, Ridge and Random Forest in three stretches of the "
        "test period: March 2017 to December 2021, the 2022-23 rate hikes, and January 2024 to August 2026.",
    )


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plain",
        type=Path,
        metavar="DIR",
        help="write versions without the figure label, title and source line to DIR",
    )
    args = parser.parse_args()
    if args.plain:
        fs.PLAIN = True
        OUT = args.plain
    frame = load()
    variants = ("wide", "mid", "narrow") if args.plain else ("wide", "narrow")
    for variant in variants:
        fs.VARIANT = variant
        narrow = variant != "wide"
        forecast_figure(frame, narrow)
        period_figure(frame, narrow)
    print(period_rmse(frame).pivot(index="period", columns="model", values="rmse").round(3))


if __name__ == "__main__":
    main()
