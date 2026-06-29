"""Monte Carlo over the REAL knockout bracket (10,000 simulations).

Starts from the actual Round-of-32 matchups (played results fixed), simulates every tie
(Poisson + Elo, extra time & penalties for draws) and advances winners through the real
bracket tree. Outputs each team's probability of reaching each round + the champion odds,
and a deterministic "most likely bracket" image.

Usage:
    python src/simulate_knockout.py
"""
import json
import math
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.default_rng(42)  # fixed seed → reproducible odds
N_SIMS = 10_000

# Round-of-32 matchups in BRACKET ORDER (top to bottom of the official bracket).
# Consecutive pairs meet in the Round of 16, and so on up the tree.
BRACKET_ORDER = [
    ("Germany", "Paraguay"), ("France", "Sweden"),
    ("South Africa", "Canada"), ("Netherlands", "Morocco"),
    ("Portugal", "Croatia"), ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"), ("Belgium", "Senegal"),
    ("Brazil", "Japan"), ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"), ("England", "DR Congo"),
    ("Argentina", "Cape Verde"), ("Australia", "Egypt"),
    ("Switzerland", "Algeria"), ("Colombia", "Ghana"),
]
ROUNDS = ["R16", "QF", "SF", "Final", "Champion"]


def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def poisson_pmf(k, lam):
    return lam ** k * math.exp(-lam) / math.factorial(k)


def main():
    elo = pd.read_csv("data/processed/elo_ratings_2026.csv").set_index("team")["elo"].to_dict()
    with open("data/processed/poisson_params.json") as f:
        P = json.load(f)["neutral"]
    bracket = pd.read_csv("data/knockout_bracket.csv")
    played = {(r.team1, r.team2): (r.home_score, r.away_score)
              for r in bracket.itertuples(index=False) if pd.notna(r.home_score)}

    def lambdas(t1, t2):
        d = (elo.get(t1, 1500) - elo.get(t2, 1500)) / 400
        return (math.exp(P["home_intercept"] + P["home_coef"] * d),
                math.exp(P["away_intercept"] + P["away_coef"] * d))

    def play_tie(t1, t2):
        """Return the winner of a single knockout tie."""
        if (t1, t2) in played:
            hs, as_ = played[(t1, t2)]
            if hs != as_:
                return t1 if hs > as_ else t2
            # real draw recorded → decided by ET/pens
        l1, l2 = lambdas(t1, t2)
        g1, g2 = rng.poisson(l1), rng.poisson(l2)
        if g1 != g2:
            return t1 if g1 > g2 else t2
        p1 = 0.5 + 0.5 * (expected_score(elo.get(t1, 1500), elo.get(t2, 1500)) - 0.5)
        return t1 if rng.random() < p1 else t2

    teams = [t for tie in BRACKET_ORDER for t in tie]
    reach = {t: {r: 0 for r in ROUNDS} for t in teams}

    for _ in range(N_SIMS):
        # Round of 32 → 16 winners
        winners = [play_tie(a, b) for a, b in BRACKET_ORDER]
        for w in winners:
            reach[w]["R16"] += 1
        # Subsequent rounds: consecutive winners meet
        for rnd in ["QF", "SF", "Final", "Champion"]:
            nxt = []
            for i in range(0, len(winners), 2):
                w = play_tie(winners[i], winners[i + 1])
                nxt.append(w)
                reach[w][rnd] += 1
            winners = nxt

    odds = pd.DataFrame(
        {t: {r: reach[t][r] / N_SIMS * 100 for r in ROUNDS} for t in teams}
    ).T.round(1)
    odds = odds.sort_values("Champion", ascending=False)
    odds.index.name = "team"
    odds.reset_index().to_csv("data/processed/knockout_odds.csv", index=False)

    print(f"Champion odds from {N_SIMS:,} simulations of the REAL bracket:\n")
    print(odds.head(12).to_string())

    # ── Deterministic "most likely bracket" (favorite advances each tie) ──
    def fav(t1, t2):
        if (t1, t2) in played:
            hs, as_ = played[(t1, t2)]
            if hs != as_:
                return t1 if hs > as_ else t2
        l1, l2 = lambdas(t1, t2)
        n = 11
        M = np.outer([poisson_pmf(i, l1) for i in range(n)], [poisson_pmf(j, l2) for j in range(n)])
        p1 = np.tril(M, -1).sum() + np.trace(M) * (0.5 + 0.5 * (expected_score(elo.get(t1,1500), elo.get(t2,1500)) - 0.5))
        return t1 if p1 >= 0.5 else t2

    path = {"R32": list(BRACKET_ORDER)}
    winners = [fav(a, b) for a, b in BRACKET_ORDER]
    for rnd in ["R16", "QF", "SF", "Final"]:
        ties = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        path[rnd] = ties
        winners = [fav(a, b) for a, b in ties]
    champion = winners[0]
    with open("data/processed/knockout_projection.json", "w", encoding="utf-8") as f:
        json.dump({"champion": champion, "path": {k: [list(t) for t in v] for k, v in path.items()}},
                  f, ensure_ascii=False, indent=1)
    print(f"\nMost likely champion: {champion}")

    # ── Champion-odds bar chart (the shareable image) ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = odds.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(top.index, top["Champion"], color="#2E7D32")
    for b, v in zip(bars, top["Champion"]):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Probability of winning the World Cup (%)")
    ax.set_title(f"World Cup 2026 — Title odds from the REAL bracket\n{N_SIMS:,} Monte Carlo simulations of the knockout stage",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("data/processed/knockout_champion_odds.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved knockout_champion_odds.png")

    # ── Bracket image (real matchups, winners of the most-likely path highlighted) ──
    draw_bracket_image(path, champion, played)
    print("Saved knockout_bracket.png")


def draw_bracket_image(path, champion, played):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = ["R32", "R16", "QF", "SF", "Final"]
    GREEN, GRAY = "#1B5E20", "#555555"
    fig, ax = plt.subplots(figsize=(15, 13))
    ax.axis("off")
    ax.set_xlim(-0.3, len(rounds) + 0.8)

    n32 = len(path["R32"])
    ax.set_ylim(-1, n32 * 1.05)
    BOX_W, BOX_H, GAP = 0.92, 0.78, 1.05

    def winner_of(tie):
        a, b = tie
        if (a, b) in played:
            hs, as_ = played[(a, b)]
            if hs != as_:
                return a if hs > as_ else b
        return None  # decided downstream; we highlight via next-round membership

    # y-position of each tie per round (centered over its feeders)
    ypos = {}
    for ri, rnd in enumerate(rounds):
        ties = path[rnd]
        for i, tie in enumerate(ties):
            if ri == 0:
                y = i * GAP
            else:
                y = (ypos[(ri - 1, 2 * i)] + ypos[(ri - 1, 2 * i + 1)]) / 2
            ypos[(ri, i)] = y

    # advancing teams per round (those that appear in the next round's ties) → highlight
    advancing = {}
    for ri in range(len(rounds)):
        nxt = set()
        if ri + 1 < len(rounds):
            for tie in path[rounds[ri + 1]]:
                nxt.update(tie)
        else:
            nxt = {champion}
        advancing[ri] = nxt

    for ri, rnd in enumerate(rounds):
        for i, tie in enumerate(path[rnd]):
            x, y = ri, ypos[(ri, i)]
            ax.add_patch(plt.Rectangle((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                                       facecolor="white", edgecolor="#999", linewidth=0.7, zorder=3))
            for k, t in enumerate(tie):
                won = t in advancing[ri]
                label = t if len(t) <= 13 else t[:12] + "."
                ax.text(x, y + 0.18 - k * 0.36, label, ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if won else "normal", color=GREEN if won else GRAY, zorder=4)
            # connector to next round
            if ri + 1 < len(rounds):
                nx, ny = ri + 1, ypos[(ri + 1, i // 2)]
                xm = (x + nx) / 2
                ax.plot([x + BOX_W / 2, xm, xm, nx - BOX_W / 2], [y, y, ny, ny],
                        color="#CCC", linewidth=0.8, zorder=1)

    fy = ypos[(len(rounds) - 1, 0)]
    ax.text(len(rounds) + 0.1, fy + 0.6, "CHAMPION", ha="center", fontsize=11, color="#666", fontweight="bold")
    ax.text(len(rounds) + 0.1, fy, champion.upper(), ha="center", fontsize=15, color=GREEN, fontweight="bold")
    for ri, rnd in enumerate(["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]):
        ax.text(ri, n32 * 1.02, rnd, ha="center", fontsize=9, fontweight="bold", color="#444")

    ax.set_title("World Cup 2026 — Projected knockout bracket (model's most likely path)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/processed/knockout_bracket.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
