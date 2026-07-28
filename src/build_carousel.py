"""Build the LinkedIn carousel PDF summarising the World Cup 2026 Predictor project.

10 portrait slides (1080x1350), design matches the project's navy/gold theme.
Output: data/processed/WorldCup2026_Recap.pdf
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.stdout.reconfigure(encoding="utf-8")

# ── Palette ──
BG     = "#0B1026"   # deep navy
CARD   = "#151B38"
STROKE = "#26305C"
GOLD   = "#FFC400"
RED    = "#C60B1E"
CELESTE= "#6CACE4"
WHITE  = "#FFFFFF"
MUTED  = "#8A93B2"
GREEN  = "#37D67A"

W, H = 108, 135  # coordinate space (portrait 4:5)


def new_slide():
    fig, ax = plt.subplots(figsize=(10.8, 13.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def footer(ax, n):
    ax.text(9, 6, "World Cup 2026 Predictor", color=MUTED, fontsize=11, va="center")
    ax.text(W - 9, 6, f"{n:02d} / 10", color=MUTED, fontsize=11, va="center", ha="right")
    ax.plot([9, W - 9], [10, 10], color=STROKE, lw=1)


def kicker(ax, text):
    ax.text(9, H - 14, text.upper(), color=GOLD, fontsize=13, fontweight="bold", va="center")
    ax.plot([9, 9 + 0.55 * len(text)], [H - 18.5, H - 18.5], color=RED, lw=3, solid_capstyle="round")


def card(ax, x, y, w, h, fc=CARD, ec=STROKE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=fc, edgecolor=ec, lw=1.5, zorder=2))


def stat(ax, x, y, big, small, color=GOLD, bigsize=34):
    ax.text(x, y, big, color=color, fontsize=bigsize, fontweight="bold", ha="center", va="center")
    ax.text(x, y - 7.5, small, color=MUTED, fontsize=11.5, ha="center", va="center")


slides = []


# ── Slide 1 — Cover ──
def s1():
    fig, ax = new_slide()
    ax.add_patch(Rectangle((0, H - 3), W, 3, facecolor=RED, edgecolor="none"))
    ax.add_patch(Rectangle((0, 0), W, 3, facecolor=GOLD, edgecolor="none"))
    ax.text(W/2, 108, "WORLD CUP 2026", color=GOLD, fontsize=22, fontweight="bold", ha="center")
    ax.text(W/2, 99, "PREDICTOR", color=WHITE, fontsize=40, fontweight="bold", ha="center")
    ax.scatter([W/2], [86], marker="*", s=1500, color=GOLD, zorder=5)
    ax.text(W/2, 74, "Can a Machine Learning model", color=WHITE, fontsize=17, ha="center")
    ax.text(W/2, 68, "out-predict a football fan?", color=WHITE, fontsize=17, ha="center")
    card(ax, 19, 40, 70, 18)
    ax.text(W/2, 51.5, "A live, end-to-end Data Science project", color=CELESTE, fontsize=13.5,
            ha="center", fontweight="bold")
    ax.text(W/2, 45, "Predictions published BEFORE every match —\nand tracked publicly, hits and misses alike.",
            color=MUTED, fontsize=11.5, ha="center", linespacing=1.5)
    ax.text(W/2, 28, "Adrián Blanco", color=WHITE, fontsize=16, fontweight="bold", ha="center")
    ax.text(W/2, 22.5, "Junior Data Scientist", color=MUTED, fontsize=12, ha="center")
    footer(ax, 1)
    return fig
slides.append(s1)


# ── Slide 2 — The idea ──
def s2():
    fig, ax = new_slide()
    kicker(ax, "The idea")
    ax.text(9, 112, "An honest forecaster,\nin public.", color=WHITE, fontsize=27, fontweight="bold",
            va="top", linespacing=1.15)
    ax.text(9, 90, "Most prediction posts appear AFTER the match.\nThis one commits before kickoff — every time.",
            color=MUTED, fontsize=13.5, va="top", linespacing=1.55)
    items = [
        ("Before", "Probabilities + an exact-score guess published\nahead of each matchday."),
        ("During", "A live dashboard updates automatically after\nevery result."),
        ("After", "Performance measured with the right metric —\nno cherry-picking."),
    ]
    y = 70
    for t, d in items:
        card(ax, 9, y - 13, 90, 12)
        ax.text(14, y - 3.5, t, color=GOLD, fontsize=15, fontweight="bold", va="center")
        ax.text(36, y - 4, d, color=WHITE, fontsize=11.5, va="center", linespacing=1.4)
        y -= 16
    footer(ax, 2)
    return fig
slides.append(s2)


# ── Slide 3 — Pipeline ──
def s3():
    fig, ax = new_slide()
    kicker(ax, "The pipeline")
    ax.text(9, 112, "A full DS pipeline,\nend to end.", color=WHITE, fontsize=27, fontweight="bold",
            va="top", linespacing=1.15)
    steps = [("DATA", "requests · BeautifulSoup"), ("SQL", "MySQL · 5 tables"),
             ("EDA", "pandas · seaborn"), ("ML", "scikit-learn"),
             ("APP", "Streamlit Cloud")]
    y = 86
    for i, (t, d) in enumerate(steps):
        card(ax, 9, y - 8.5, 90, 8, fc=CARD)
        ax.add_patch(FancyBboxPatch((11, y - 7), 20, 5, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=RED if i % 2 == 0 else CELESTE, edgecolor="none"))
        ax.text(21, y - 4.5, t, color=WHITE, fontsize=13, fontweight="bold", ha="center", va="center")
        ax.text(35, y - 4.5, d, color=MUTED, fontsize=12, va="center")
        if i < len(steps) - 1:
            ax.text(54, y - 10.7, "▼", color=STROKE, fontsize=11, ha="center", va="center")
        y -= 11.5
    card(ax, 9, 20, 90, 12, fc=CARD)
    ax.text(W/2, 28.5, "49,477 international matches  ·  1,246 players", color=GOLD,
            fontsize=13.5, fontweight="bold", ha="center")
    ax.text(W/2, 23.5, "the raw material behind every prediction", color=MUTED, fontsize=11, ha="center")
    footer(ax, 3)
    return fig
slides.append(s3)


# ── Slide 4 — Models ──
def s4():
    fig, ax = new_slide()
    kicker(ax, "The models")
    ax.text(9, 112, "Four models,\none ensemble.", color=WHITE, fontsize=27, fontweight="bold",
            va="top", linespacing=1.15)
    models = [
        ("Elo ratings", "Built from scratch over 49k matches. Updates\nwith every result — the model keeps learning."),
        ("Random Forest", "Match outcome (win / draw / win), using Elo\nas a live feature."),
        ("Poisson", "Exact-score model — goals from the Elo gap."),
        ("Monte Carlo", "10,000 simulations of the tournament to get\nchampion odds."),
    ]
    y = 88
    for t, d in models:
        card(ax, 9, y - 12, 90, 11)
        ax.text(14, y - 3.5, t, color=GOLD, fontsize=15, fontweight="bold", va="center")
        ax.text(14, y - 8.5, d, color=WHITE, fontsize=11, va="center", linespacing=1.4)
        y -= 14.5
    ax.text(W/2, 22, "+ an RF·Poisson ensemble that wins on RPS", color=CELESTE,
            fontsize=12.5, fontweight="bold", ha="center", style="italic")
    footer(ax, 4)
    return fig
slides.append(s4)


# ── Slide 5 — Honest metrics ──
def s5():
    fig, ax = new_slide()
    kicker(ax, "Measured honestly")
    ax.text(9, 112, "Accuracy lies.\nSo I used RPS.", color=WHITE, fontsize=27, fontweight="bold",
            va="top", linespacing=1.15)
    ax.text(9, 90, "For a probabilistic forecaster of ordered outcomes,\naccuracy is the wrong yardstick. The Ranked\nProbability Score (RPS) is the standard metric.",
            color=MUTED, fontsize=13, va="top", linespacing=1.55)
    card(ax, 9, 48, 43, 22)
    stat(ax, 30.5, 63, "0.188", "RPS  (≈0.20 is on par)", color=GOLD, bigsize=32)
    card(ax, 56, 48, 43, 22)
    stat(ax, 77.5, 63, "0.031", "calibration error (ECE)", color=GREEN, bigsize=32)
    ax.text(W/2, 38, "Well-calibrated: when the model says 70%,\nit happens about 70% of the time.",
            color=WHITE, fontsize=12.5, ha="center", linespacing=1.5)
    footer(ax, 5)
    return fig
slides.append(s5)


# ── Slide 6 — Group stage ──
def s6():
    fig, ax = new_slide()
    kicker(ax, "Results · group stage")
    ax.text(9, 112, "72 matches.\nThe warm-up.", color=WHITE, fontsize=27, fontweight="bold",
            va="top", linespacing=1.15)
    card(ax, 9, 74, 43, 20); stat(ax, 30.5, 87, "53%", "outcomes correct", bigsize=32)
    card(ax, 56, 74, 43, 20); stat(ax, 77.5, 87, "14", "exact scores nailed", color=CELESTE, bigsize=32)
    ax.text(9, 63, "The exact-score model hit 19% of scorelines —\nwell above the ~11% of world-class models.",
            color=MUTED, fontsize=12.5, va="top", linespacing=1.5)
    card(ax, 9, 26, 90, 22, fc=CARD)
    ax.text(W/2, 43, "The accuracy trap", color=GOLD, fontsize=14, fontweight="bold", ha="center")
    ax.text(W/2, 34, "Raw accuracy looked noisy over a small sample,\nbut RPS stayed right on its historical value.\nThe model behaved exactly as expected.",
            color=WHITE, fontsize=11.5, ha="center", linespacing=1.5)
    footer(ax, 6)
    return fig
slides.append(s6)


# ── Slide 7 — Knockout ──
def s7():
    fig, ax = new_slide()
    kicker(ax, "Results · knockouts")
    ax.text(9, 112, "Where it got\nserious.", color=WHITE, fontsize=27, fontweight="bold",
            va="top", linespacing=1.15)
    card(ax, 9, 76, 90, 20, fc=CARD, ec=GOLD)
    stat(ax, 30, 89, "25/31", "ties called right", bigsize=30)
    stat(ax, 77, 89, "81%", "knockout accuracy", bigsize=30)
    ax.plot([54, 54], [79, 93], color=STROKE, lw=1.2)
    ax.text(9, 66, "And where it missed, it missed for the one\nreason nobody can beat:",
            color=MUTED, fontsize=13, va="top", linespacing=1.5)
    card(ax, 9, 30, 90, 24)
    ax.text(W/2, 48, "4 of the misses were penalty shootouts", color=GOLD,
            fontsize=14, fontweight="bold", ha="center")
    ax.text(W/2, 38, "A shootout is a coin flip. The model treats it\nas ~50/50 instead of pretending to know —\nbecause faking it would be dishonest.",
            color=WHITE, fontsize=11.5, ha="center", linespacing=1.5)
    footer(ax, 7)
    return fig
slides.append(s7)


# ── Slide 8 — The champion (headline) ──
def s8():
    fig, ax = new_slide()
    ax.add_patch(Rectangle((0, H - 3), W, 3, facecolor=GOLD, edgecolor="none"))
    kicker(ax, "The headline")
    ax.text(W/2, 108, "It called the champion.", color=WHITE, fontsize=25, fontweight="bold", ha="center")
    ax.scatter([W/2], [95], marker="*", s=1300, color=GOLD, zorder=5)
    ax.text(W/2, 84, "SPAIN", color=WHITE, fontsize=40, fontweight="bold", ha="center")
    ax.text(W/2, 76, "WORLD CHAMPIONS", color=GOLD, fontsize=17, fontweight="bold", ha="center")
    ax.plot([38, 70], [71, 71], color=RED, lw=4, solid_capstyle="round")
    ax.text(W/2, 63, "Before the final, the model's odds:", color=MUTED, fontsize=12.5, ha="center")
    ax.text(34, 53, "52%", color=GOLD, fontsize=34, fontweight="bold", ha="center")
    ax.text(34, 46, "Spain", color=WHITE, fontsize=13, ha="center")
    ax.text(54, 51, "vs", color=MUTED, fontsize=15, ha="center", style="italic")
    ax.text(74, 53, "48%", color=CELESTE, fontsize=34, fontweight="bold", ha="center")
    ax.text(74, 46, "Argentina", color=WHITE, fontsize=13, ha="center")
    card(ax, 9, 22, 90, 15)
    ax.text(W/2, 31.5, "Final: Spain 1 – 0 Argentina (a.e.t.)", color=WHITE, fontsize=13.5,
            fontweight="bold", ha="center")
    ax.text(W/2, 26, "The dream final it had flagged since the round of 32.", color=MUTED,
            fontsize=11.5, ha="center")
    footer(ax, 8)
    return fig
slides.append(s8)


# ── Slide 9 — Whole tournament ──
def s9():
    fig, ax = new_slide()
    kicker(ax, "The full picture")
    ax.text(9, 112, "The whole\ntournament.", color=WHITE, fontsize=27, fontweight="bold",
            va="top", linespacing=1.15)
    data = [("103", "matches predicted"), ("60%", "overall accuracy"),
            ("18", "exact scores"), ("✓", "champion called")]
    pos = [(30.5, 78), (77.5, 78), (30.5, 50), (77.5, 50)]
    for (big, small), (x, y) in zip(data, pos):
        card(ax, x - 21.5, y - 10, 43, 20)
        col = GREEN if big == "✓" else GOLD
        stat(ax, x, y + 3, big, small, color=col, bigsize=34 if big != "✓" else 30)
    ax.text(W/2, 28, "A model that stayed honest all tournament —\nand still got the big one right.",
            color=WHITE, fontsize=12.5, ha="center", linespacing=1.5)
    footer(ax, 9)
    return fig
slides.append(s9)


# ── Slide 10 — Closing / CTA ──
def s10():
    fig, ax = new_slide()
    ax.add_patch(Rectangle((0, 0), W, 3, facecolor=GOLD, edgecolor="none"))
    ax.text(W/2, 116, "Let's talk.", color=WHITE, fontsize=34, fontweight="bold", ha="center")
    ax.text(W/2, 106, "Open to Junior Data Scientist roles", color=GOLD, fontsize=15,
            fontweight="bold", ha="center")
    card(ax, 9, 74, 90, 24)
    ax.text(W/2, 93, "Built with", color=MUTED, fontsize=11.5, ha="center")
    ax.text(W/2, 86, "Python · pandas · scikit-learn · MySQL", color=WHITE, fontsize=13.5,
            fontweight="bold", ha="center")
    ax.text(W/2, 79.5, "Streamlit · matplotlib · soccerdata", color=WHITE, fontsize=13.5,
            fontweight="bold", ha="center")
    ax.text(W/2, 62, "See it live", color=MUTED, fontsize=12, ha="center")
    ax.text(W/2, 55.5, "predictor-worldcup2026.streamlit.app", color=CELESTE, fontsize=14,
            fontweight="bold", ha="center")
    ax.text(W/2, 46, "Code", color=MUTED, fontsize=12, ha="center")
    ax.text(W/2, 39.5, "github.com/AdriBlanco0/worldcup2026-predictor", color=CELESTE,
            fontsize=13, fontweight="bold", ha="center")
    ax.text(W/2, 26, "Adrián Blanco", color=WHITE, fontsize=17, fontweight="bold", ha="center")
    ax.text(W/2, 20, "linkedin.com/in/adrianblancoajenjo", color=MUTED, fontsize=12, ha="center")
    return fig
slides.append(s10)


def main():
    out = "data/processed/WorldCup2026_Recap.pdf"
    with PdfPages(out) as pdf:
        for s in slides:
            fig = s()
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f"Saved {out} ({len(slides)} slides)")


if __name__ == "__main__":
    main()
