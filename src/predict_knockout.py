"""Predict knockout ties (single match, no draws — extra time + penalties decide).

For each tie: P(team1 advances) = P(win in 90') + P(draw) * (Elo-weighted shootout, dampened
towards 50/50). Also gives the most likely 90-minute scoreline from the Poisson model.

Usage:
    python src/predict_knockout.py
"""
import json
import math
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BRACKET = "data/knockout_bracket.csv"
OUT = "data/processed/knockout_predictions.csv"


def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def poisson_pmf(k, lam):
    return lam ** k * math.exp(-lam) / math.factorial(k)


def main():
    elo = pd.read_csv("data/processed/elo_ratings_2026.csv").set_index("team")["elo"].to_dict()
    with open("data/processed/poisson_params.json") as f:
        params = json.load(f)["neutral"]  # knockouts on neutral ground

    def matrix(t1, t2, n=11):
        d = (elo.get(t1, 1500) - elo.get(t2, 1500)) / 400
        l1 = math.exp(params["home_intercept"] + params["home_coef"] * d)
        l2 = math.exp(params["away_intercept"] + params["away_coef"] * d)
        M = np.outer([poisson_pmf(i, l1) for i in range(n)], [poisson_pmf(j, l2) for j in range(n)])
        return M, l1, l2

    bracket = pd.read_csv(BRACKET)
    rows = []
    for m in bracket.itertuples(index=False):
        t1, t2 = m.team1, m.team2
        M, l1, l2 = matrix(t1, t2)
        p1_90 = np.tril(M, -1).sum()
        pdraw = np.trace(M)
        p2_90 = np.triu(M, 1).sum()
        # Shootout / extra time: Elo edge dampened halfway to 50/50
        share = expected_score(elo.get(t1, 1500), elo.get(t2, 1500))
        p1_adv = p1_90 + pdraw * (0.5 + 0.5 * (share - 0.5))
        # Most likely 90' scoreline
        i, j = np.unravel_index(M.argmax(), M.shape)

        played = pd.notna(m.home_score)
        rows.append({
            "match_id": m.match_id, "round": m.round, "team1": t1, "team2": t2,
            "kickoff_spain": m.kickoff_spain,
            "p_team1_adv": round(p1_adv * 100, 1), "p_team2_adv": round((1 - p1_adv) * 100, 1),
            "pred_score": f"{i}-{j}",
            "favorite": t1 if p1_adv >= 0.5 else t2,
            "home_score": int(m.home_score) if played else None,
            "away_score": int(m.away_score) if played else None,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("kickoff_spain")
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} knockout predictions -> {OUT}\n")
    for r in df.itertuples(index=False):
        played = ""
        if pd.notna(r.home_score):
            res = "✅" if ((r.home_score > r.away_score and r.favorite == r.team1) or
                          (r.away_score > r.home_score and r.favorite == r.team2)) else "❌"
            played = f"  [played {r.home_score}-{r.away_score} {res}]"
        print(f"{r.team1} vs {r.team2}: {r.favorite} advances {max(r.p_team1_adv, r.p_team2_adv):.0f}% "
              f"(pred {r.pred_score}){played}")


if __name__ == "__main__":
    main()
