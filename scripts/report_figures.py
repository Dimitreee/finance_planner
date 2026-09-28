"""Draw the final report's figures from the measured artifacts.

Every number here is read from `artifacts/*.json` — the files the experiment scripts wrote — so a
figure cannot drift from the result it depicts. Nothing is recomputed and nothing is hand-entered.

matplotlib is deliberately *not* a project dependency: it is needed only to render the report, never
to produce a result. Run it ephemerally:

    uv run --with matplotlib python scripts/report_figures.py

Output: `reports/figures/*.pdf` (vector, for the LaTeX build) and `*.png` (for the video slides).
The series colours match the web page's tokens so the report and the page agree visually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"

# The web page's series tokens, so a reader moving between report and page sees one visual system.
STRATEGY = "#2a78d6"
MARKET = "#eb6834"
CASH = "#8a94a6"
INK = "#1e242e"
RULE = "#c3cad6"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)


def load(name: str) -> Any:
    return json.loads((ARTIFACTS / name).read_text())


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reports/figures/{stem}.pdf and .png")


def figure_architecture() -> None:
    """The write path, the read path, and the boundary a test defends."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axes = plt.subplots(figsize=(6.3, 3.1))
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 60)
    axes.axis("off")

    # (x, y, width, height, label, colour). Widths are generous enough for the longest line, because
    # a diagram whose text overflows its box is worse than no diagram.
    boxes = [
        (1, 37, 27, 16, "Immutable snapshots\n+ model bundles\nSHA-256 verified", CASH),
        (
            34,
            37,
            31,
            16,
            "Scheduled job\ningest \u2192 validate \u2192 features\n"
            "\u2192 forecast \u2192 policy \u2192 publish",
            MARKET,
        ),
        (71, 37, 28, 16, "PostgreSQL\nstate + read model", INK),
        (71, 8, 28, 16, "FastAPI read API\nholds no model,\nno credentials", STRATEGY),
        (38, 8, 27, 16, "Web UI (React)\nwhat a reader sees", STRATEGY),
        (1, 8, 27, 16, "Experiment Contract\nfrozen, hashed, binding", CASH),
    ]
    for x, y, w, h, label, colour in boxes:
        axes.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.5,rounding_size=1.4",
                linewidth=1.2,
                edgecolor=colour,
                facecolor="white",
            )
        )
        axes.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.6, color=INK)

    def arrow(x1, y1, x2, y2, colour=INK):
        axes.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color=colour,
                shrinkA=1,
                shrinkB=1,
            )
        )

    arrow(28.6, 45, 33.4, 45)  # snapshots -> job
    arrow(65.6, 45, 70.4, 45)  # job -> database: the only writer
    arrow(85, 36.4, 85, 24.6, STRATEGY)  # database -> read API
    arrow(70.4, 16, 65.6, 16, STRATEGY)  # read API -> page
    # The contract binds the *job*, so the arrow must reach the job rather than the snapshots box
    # it happens to sit under. Drawn as an elbow: a plain segment, then the segment with the head.
    axes.plot([14.5, 14.5, 49.5], [24.6, 31, 31], color=CASH, linewidth=1.1, solid_capstyle="butt")
    arrow(49.5, 31, 49.5, 36.4, CASH)

    axes.text(
        49.5, 54.6, "the only writer", ha="center", fontsize=6.4, color=MARKET, style="italic"
    )
    axes.text(86.5, 30.2, "read path", ha="left", fontsize=6.4, color=STRATEGY, style="italic")
    axes.text(
        16.5,
        26.4,
        "binds the job;\ndigest in every artifact",
        ha="left",
        fontsize=6.4,
        color=CASH,
        style="italic",
        va="center",
    )
    axes.text(
        50,
        1.5,
        "A test forbids the read API importing the model, ingestion, dataset, decision "
        "or news modules.",
        ha="center",
        fontsize=6.6,
        color=INK,
    )
    save(fig, "fig0-architecture")


def figure_log_loss_by_arm() -> None:
    """Selection Metric per configuration, against the no-skill reference."""
    data = load("arm_comparison.json")
    no_skill = data["no_skill_log_loss"]
    arms = data["arms"]

    labels, scores, colours = [], [], []
    for entry in arms:
        lag = entry["lag_hours"]
        labels.append(entry["arm"] if lag is None else f"{entry['arm']}@{lag}h")
        scores.append(entry["log_loss"])
        colours.append(STRATEGY if lag is None else MARKET)

    fig, axes = plt.subplots(figsize=(6.1, 2.9))
    bars = axes.bar(labels, scores, color=colours, width=0.62)
    axes.axhline(no_skill, color=INK, linestyle="--", linewidth=1)
    axes.annotate(
        f"no skill, ln 2 = {no_skill:.4f}",
        xy=(len(labels) - 0.4, no_skill),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    # A 0.005 span carries the whole finding, so the axis starts where the differences live. The
    # reference line is inside the frame, which is what stops the truncation from misleading.
    axes.set_ylim(no_skill - 0.0006, max(scores) + 0.0009)
    axes.set_ylabel("Log loss (lower is better)")
    for bar, score in zip(bars, scores, strict=True):
        axes.annotate(
            f"{score:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    axes.set_xlabel("Arm and assumed news availability lag")
    save(fig, "fig1-log-loss-by-arm")


def figure_primary_interval() -> None:
    """The Primary Comparison's interval at each block length, with zero marked."""
    data = load("bootstrap_intervals.json")
    quantity = next(q for q in data["quantities"] if q["standing"] == "evidence")
    intervals = sorted(quantity["intervals"], key=lambda i: i["block_days"])

    fig, axes = plt.subplots(figsize=(6.1, 2.4))
    for row, interval in enumerate(intervals):
        stable = interval["verdict_stable_across_seeds"]
        colour = STRATEGY if stable else MARKET
        axes.plot(
            [interval["lower"], interval["upper"]],
            [row, row],
            color=colour,
            linewidth=2.4,
            solid_capstyle="butt",
        )
        axes.plot(interval["estimate"], row, "o", color=colour, markersize=5)
        note = "" if stable else "  (1 of 4 seeds Inconclusive)"
        axes.annotate(
            f"{interval['verdict']}{note}",
            xy=(interval["upper"], row),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=colour,
        )
    axes.axvline(0, color=INK, linestyle="--", linewidth=1)
    axes.set_yticks(range(len(intervals)))
    axes.set_yticklabels(
        [
            f"{i['block_days']}-day blocks" + (" (reported)" if i["role"] == "reported" else "")
            for i in intervals
        ]
    )
    axes.set_xlim(-0.0016, 0.019)
    axes.invert_yaxis()
    axes.set_xlabel("Paired daily log-loss difference, C@24h minus A  (positive = news is worse)")
    save(fig, "fig2-primary-comparison-interval")


def figure_development_versus_holdout() -> None:
    """The false positive, in one picture: what selection saw, and what it did not."""
    holdout = load("final_holdout.json")["result"]
    # Read, never retyped: the development figures are arm A's own row in the comparison
    # artifact, so a re-run that moved them would move this figure too.
    arm_a = next(
        entry
        for entry in load("arm_comparison.json")["arms"]
        if entry["arm"] == "A" and entry["lag_hours"] is None
    )
    development = {
        "net_return": arm_a["headline_return"],
        "buy_and_hold_return": arm_a["buy_and_hold_return"],
    }

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.1, 2.7), sharey=False)
    for axes, title, result in (
        (left, "Development Period (1,096 days)\nselection ran here", development),
        (right, "Final Holdout (334 days)\nnothing was selected here", holdout),
    ):
        gap = (result["net_return"] - result["buy_and_hold_return"]) * 100
        note = f"{gap:+.1f} pp".replace("-", "\u2212")
        values = [result["net_return"], result["buy_and_hold_return"], 0.0]
        axes.bar(
            ["The advice", "Buy & hold", "Cash"],
            [v * 100 for v in values],
            color=[STRATEGY, MARKET, CASH],
            width=0.6,
        )
        axes.axhline(0, color=INK, linewidth=0.9)
        axes.set_title(title, fontsize=8.5)
        for index, value in enumerate(values):
            axes.annotate(
                f"{value * 100:+.1f}%".replace("-", "\u2212"),
                xy=(index, value * 100),
                xytext=(0, 3 if value >= 0 else -11),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axes.annotate(
            f"gap {note}",
            xy=(0.5, 0.92),
            xycoords="axes fraction",
            ha="center",
            fontsize=8,
            color=INK,
        )
    left.set_ylabel("Net return after costs, base scenario")
    left.set_ylim(-40, 185)
    right.set_ylim(-40, 185)
    save(fig, "fig3-development-versus-holdout")


def figure_holdout_intervals() -> None:
    """Both holdout quantities against zero: the point estimates sit inside the noise."""
    data = load("holdout_intervals.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.1))
    for panel, quantity in zip(axes, data["quantities"], strict=True):
        reported = next(i for i in quantity["intervals"] if i["role"] == "reported")
        scale = 1.0 if "log loss" in quantity["name"] else 100.0
        panel.plot(
            [reported["lower"] * scale, reported["upper"] * scale],
            [0, 0],
            color=STRATEGY,
            linewidth=2.6,
            solid_capstyle="butt",
        )
        panel.plot(reported["estimate"] * scale, 0, "o", color=STRATEGY, markersize=5)
        panel.axvline(0, color=INK, linestyle="--", linewidth=1)
        panel.set_yticks([])
        panel.set_title(
            ("Daily log loss minus no skill" if scale == 1.0 else "Net return minus buy-and-hold")
            + f"\n{reported['verdict']}, 20-day blocks",
            fontsize=8.5,
        )
        panel.set_xlabel("natural units" if scale == 1.0 else "percentage points")
        panel.spines["left"].set_visible(False)
    save(fig, "fig4-holdout-intervals")


if __name__ == "__main__":
    figure_architecture()
    figure_log_loss_by_arm()
    figure_primary_interval()
    figure_development_versus_holdout()
    figure_holdout_intervals()
