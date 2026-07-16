"""Generate the shareable 'Final prediction' graphic for LinkedIn.

Spain vs Argentina — head-to-head title odds + the model's knockout track record.
Outputs data/processed/final_prediction.png (1200x1200, LinkedIn-friendly square).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

# ── Data ──
P_SPAIN, P_ARG = 51.9, 48.1
RECORD = "24/30"
RECORD_PCT = "80%"

# ── Colours ──
BG = "#0B1026"          # deep navy
SPAIN = "#C60B1E"       # rojo España
SPAIN_Y = "#FFC400"     # amarillo
ARG = "#6CACE4"         # celeste Argentina
WHITE = "#FFFFFF"
MUTED = "#8A93B2"

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ── Header ──
ax.text(50, 94, "MUNDIAL 2026 · LA FINAL", ha="center", va="center",
        color=SPAIN_Y, fontsize=17, fontweight="bold", family="DejaVu Sans")
ax.text(50, 88.5, "Predicción del modelo de Machine Learning",
        ha="center", va="center", color=MUTED, fontsize=12.5)

# ── Teams ──
ax.text(27, 76, "ESPAÑA", ha="center", va="center", color=WHITE, fontsize=30, fontweight="bold")
ax.text(73, 76, "ARGENTINA", ha="center", va="center", color=WHITE, fontsize=27, fontweight="bold")
ax.text(50, 76, "vs", ha="center", va="center", color=MUTED, fontsize=18, style="italic")

# Team accent underlines
ax.plot([17, 37], [70.5, 70.5], color=SPAIN, lw=5, solid_capstyle="round")
ax.plot([61, 85], [70.5, 70.5], color=ARG, lw=5, solid_capstyle="round")

# ── Probabilities ──
ax.text(27, 60, f"{P_SPAIN:.0f}%", ha="center", va="center", color=SPAIN_Y, fontsize=58, fontweight="bold")
ax.text(73, 60, f"{P_ARG:.0f}%", ha="center", va="center", color=ARG, fontsize=58, fontweight="bold")
ax.text(50, 60, "—", ha="center", va="center", color=MUTED, fontsize=30)
ax.text(50, 51.5, "probabilidad de ser campeón (10.000 simulaciones Monte Carlo)",
        ha="center", va="center", color=MUTED, fontsize=10.5)

# ── Split bar ──
bx, bw, by, bh = 12, 76, 42, 4.2
sw = bw * P_SPAIN / 100
ax.add_patch(FancyBboxPatch((bx, by), sw, bh, boxstyle="round,pad=0,rounding_size=0.6",
                            facecolor=SPAIN, edgecolor="none"))
ax.add_patch(FancyBboxPatch((bx + sw, by), bw - sw, bh, boxstyle="round,pad=0,rounding_size=0.6",
                            facecolor=ARG, edgecolor="none"))

# ── Track record panel ──
ax.add_patch(FancyBboxPatch((14, 20), 72, 20, boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor="#151B38", edgecolor="#26305C", lw=1.5))
ax.text(50, 34.5, "El modelo en la fase eliminatoria", ha="center", va="center",
        color=WHITE, fontsize=13, fontweight="bold")
ax.text(37, 27, RECORD, ha="center", va="center", color=SPAIN_Y, fontsize=30, fontweight="bold")
ax.text(37, 22.5, "cruces acertados", ha="center", va="center", color=MUTED, fontsize=10)
ax.text(63, 27, RECORD_PCT, ha="center", va="center", color=SPAIN_Y, fontsize=30, fontweight="bold")
ax.text(63, 22.5, "de acierto", ha="center", va="center", color=MUTED, fontsize=10)
ax.plot([50, 50], [22, 32], color="#26305C", lw=1.2)

# ── Footer ──
ax.text(50, 11, "Predicciones publicadas ANTES de cada partido", ha="center", va="center",
        color=MUTED, fontsize=11, style="italic")
ax.text(50, 5.5, "predictor-worldcup2026.streamlit.app", ha="center", va="center",
        color=WHITE, fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("data/processed/final_prediction.png", dpi=120, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("Saved data/processed/final_prediction.png")
