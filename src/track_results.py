"""Track real results against model predictions.

Workflow during the tournament:
1. Fill in real scores in data/results_2026.csv (only home_score / away_score columns)
2. Run:  python src/track_results.py
3. The script prints per-match hits/misses and overall accuracy for BOTH models
   (v0 Random Forest and v1 Elo) and saves data/processed/model_performance.csv
"""

import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RESULTS = "data/results_2026.csv"
PRED_V0 = "data/processed/predictions_2026_group_stage_v0.csv"
PRED_V1 = "data/processed/predictions_2026_group_stage_v1.csv"
OUTPUT = "data/processed/model_performance.csv"


def actual_outcome(row):
    if row["home_score"] > row["away_score"]:
        return "Home win"
    if row["home_score"] < row["away_score"]:
        return "Away win"
    return "Draw"


def main():
    results = pd.read_csv(RESULTS)
    played = results[results["home_score"].notna()].copy()

    if len(played) == 0:
        print("No results recorded yet. Fill in scores in data/results_2026.csv and re-run.")
        return

    played["actual"] = played.apply(actual_outcome, axis=1)

    # Bring in each model's pick and probabilities
    v0 = pd.read_csv(PRED_V0)[["home_team", "away_team", "prediction", "p_home_win", "p_draw", "p_away_win"]]
    v0.columns = ["home_team", "away_team", "v0_pick", "v0_p_home", "v0_p_draw", "v0_p_away"]
    v1 = pd.read_csv(PRED_V1)[["home_team", "away_team", "prediction"]]
    v1.columns = ["home_team", "away_team", "v1_pick"]

    df = played.merge(v0, on=["home_team", "away_team"], how="left")
    df = df.merge(v1, on=["home_team", "away_team"], how="left")

    df["v0_correct"] = df["v0_pick"] == df["actual"]
    df["v1_correct"] = df["v1_pick"] == df["actual"]

    # ── Per-match report ──────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"MODEL PERFORMANCE — {len(df)} matches played")
    print(f"{'='*72}")
    for _, m in df.iterrows():
        score = f"{int(m['home_score'])}-{int(m['away_score'])}"
        v0_mark = "✅" if m["v0_correct"] else "❌"
        v1_mark = "✅" if m["v1_correct"] else "❌"
        print(f"{m['home_team']} {score} {m['away_team']:<18} | real: {m['actual']:<8} "
              f"| v0: {m['v0_pick']:<8} {v0_mark} | v1: {m['v1_pick']:<8} {v1_mark}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"v0 (Random Forest): {df['v0_correct'].sum()}/{len(df)} correct "
          f"({df['v0_correct'].mean()*100:.1f}%)")
    print(f"v1 (Elo):           {df['v1_correct'].sum()}/{len(df)} correct "
          f"({df['v1_correct'].mean()*100:.1f}%)")
    print(f"{'='*72}")

    # ── Markdown table for the README ─────────────────────────────────
    print("\nREADME table (copy-paste):\n")
    print("| Matches played | v0 correct | v0 accuracy | v1 correct | v1 accuracy |")
    print("|---|---|---|---|---|")
    print(f"| {len(df)} | {df['v0_correct'].sum()} | {df['v0_correct'].mean()*100:.1f}% "
          f"| {df['v1_correct'].sum()} | {df['v1_correct'].mean()*100:.1f}% |")

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
