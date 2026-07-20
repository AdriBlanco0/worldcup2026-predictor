"""Generate the 'Champion' celebration graphic for LinkedIn.

Spain world champions — the model called it (Spain was the pre-final favourite).
Outputs data/processed/champion.png (1200x1200 square).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

BG = "#0B1026"
GOLD = "#FFC400"
SPAIN = "#C60B1E"
WHITE = "#FFFFFF"
MUTED = "#8A93B2"

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Header
ax.text(50, 92, "MUNDIAL 2026", ha="center", va="center", color=MUTED, fontsize=15, fontweight="bold")

# Trophy + champion
ax.scatter([50], [80], marker="*", s=4200, color=GOLD, edgecolor="none", zorder=5)
ax.text(50, 66, "ESPAÑA", ha="center", va="center", color=WHITE, fontsize=52, fontweight="bold")
ax.text(50, 58.5, "CAMPEONA DEL MUNDO", ha="center", va="center", color=GOLD, fontsize=22, fontweight="bold")
ax.plot([28, 72], [53, 53], color=SPAIN, lw=5, solid_capstyle="round")

# Final result
ax.text(50, 46, "Final:  España 1 – 0 Argentina  (prórroga)", ha="center", va="center",
        color=MUTED, fontsize=14, style="italic")

# The call-out: the model nailed it
ax.add_patch(FancyBboxPatch((13, 22), 74, 18, boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor="#151B38", edgecolor="#26305C", lw=1.5))
ax.text(50, 34.5, "El modelo la tenía como favorita al título (52%)",
        ha="center", va="center", color=WHITE, fontsize=13, fontweight="bold")
ax.text(37, 28, "25/31", ha="center", va="center", color=GOLD, fontsize=27, fontweight="bold")
ax.text(37, 24, "cruces acertados", ha="center", va="center", color=MUTED, fontsize=9.5)
ax.text(63, 28, "81%", ha="center", va="center", color=GOLD, fontsize=27, fontweight="bold")
ax.text(63, 24, "acierto en eliminatorias", ha="center", va="center", color=MUTED, fontsize=9.5)
ax.plot([50, 50], [23.5, 32], color="#26305C", lw=1.2)

# Footer
ax.text(50, 13, "Un proyecto de Data Science en directo · predicciones antes de cada partido",
        ha="center", va="center", color=MUTED, fontsize=10, style="italic")
ax.text(50, 7.5, "predictor-worldcup2026.streamlit.app", ha="center", va="center",
        color=WHITE, fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("data/processed/champion.png", dpi=120, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("Saved data/processed/champion.png")
