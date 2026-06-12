"""Train model v1 (Random Forest + Elo) and persist it to models/rf_v1.joblib.

Run once (or whenever the training logic changes):
    python src/train_model.py
"""

import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.stdout.reconfigure(encoding="utf-8")

HEIR_NATIONS = {
    "West Germany": "Germany", "Soviet Union": "Russia", "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia", "Czechoslovakia": "Czech Republic",
    "Zaire": "DR Congo", "Dutch East Indies": "Indonesia",
}

FEATURES = [
    "home_wc_played", "home_win_rate", "home_goals_for", "home_goals_against",
    "home_recent_form", "home_is_host", "away_wc_played", "away_win_rate",
    "away_goals_for", "away_goals_against", "away_recent_form", "away_is_host",
    "diff_win_rate", "diff_form", "elo_home", "elo_away", "elo_diff",
]


def load_wc_matches():
    matches = pd.read_csv("data/raw/historical/matches.csv")
    m = matches[matches["tournament_name"].str.contains("Men's")].copy()
    m["year"] = m["tournament_name"].str[:4].astype(int)
    m["match_date"] = pd.to_datetime(m["match_date"])
    m["home_team_name"] = m["home_team_name"].replace(HEIR_NATIONS)
    m["away_team_name"] = m["away_team_name"].replace(HEIR_NATIONS)
    return m


def build_team_matches(matches_m):
    """Long format: one row per team per match."""
    home = matches_m[["year", "match_id", "home_team_name", "home_team_score",
                      "away_team_score", "home_team_win", "draw"]].copy()
    home.columns = ["year", "match_id", "team", "goals_for", "goals_against", "won", "draw"]
    away = matches_m[["year", "match_id", "away_team_name", "away_team_score",
                      "home_team_score", "away_team_win", "draw"]].copy()
    away.columns = ["year", "match_id", "team", "goals_for", "goals_against", "won", "draw"]
    return pd.concat([home, away], ignore_index=True)


def team_features(team_matches, team, year, prefix):
    """Historical features using ONLY matches before `year` (no leakage)."""
    past = team_matches[(team_matches["team"] == team) & (team_matches["year"] < year)]
    if len(past) == 0:
        return {f"{prefix}_wc_played": 0, f"{prefix}_win_rate": 0.0, f"{prefix}_goals_for": 0.0,
                f"{prefix}_goals_against": 0.0, f"{prefix}_recent_form": 0.0}
    last_two = sorted(past["year"].unique())[-2:]
    recent = past[past["year"].isin(last_two)]
    return {
        f"{prefix}_wc_played": past["year"].nunique(),
        f"{prefix}_win_rate": past["won"].mean(),
        f"{prefix}_goals_for": past["goals_for"].mean(),
        f"{prefix}_goals_against": past["goals_against"].mean(),
        f"{prefix}_recent_form": recent["won"].mean(),
    }


def main():
    matches_m = load_wc_matches()
    team_matches = build_team_matches(matches_m)

    # Pre-match Elo from the internationals artifact (computed in notebook 03)
    intl = pd.read_csv("data/processed/internationals_with_elo.csv")
    intl["date"] = pd.to_datetime(intl["date"])
    elo_cols = intl[["date", "home_team", "away_team", "elo_home_pre", "elo_away_pre"]]

    merged = matches_m.merge(
        elo_cols, left_on=["match_date", "home_team_name", "away_team_name"],
        right_on=["date", "home_team", "away_team"], how="left",
    )
    swapped = matches_m.merge(
        elo_cols, left_on=["match_date", "home_team_name", "away_team_name"],
        right_on=["date", "away_team", "home_team"], how="left",
    )
    merged["elo_home_pre"] = merged["elo_home_pre"].fillna(swapped["elo_away_pre"])
    merged["elo_away_pre"] = merged["elo_away_pre"].fillna(swapped["elo_home_pre"])

    tournaments = pd.read_csv("data/raw/historical/tournaments.csv")
    tournaments_m = tournaments[tournaments["tournament_name"].str.contains("Men's")].copy()
    tournaments_m["year"] = tournaments_m["tournament_name"].str[:4].astype(int)
    host_by_year = tournaments_m.set_index("year")["host_country"].to_dict()

    train_matches = merged[(merged["year"] >= 1962) & (merged["group_stage"] == 1) &
                           (merged["elo_home_pre"].notna())].copy()

    rows = []
    for _, m in train_matches.iterrows():
        row = {}
        row.update(team_features(team_matches, m["home_team_name"], m["year"], "home"))
        row.update(team_features(team_matches, m["away_team_name"], m["year"], "away"))
        host = host_by_year.get(m["year"], "")
        row["home_is_host"] = int(m["home_team_name"] == host)
        row["away_is_host"] = int(m["away_team_name"] == host)
        row["elo_home"] = m["elo_home_pre"]
        row["elo_away"] = m["elo_away_pre"]
        row["elo_diff"] = m["elo_home_pre"] - m["elo_away_pre"]
        row["target"] = 0 if m["home_team_win"] == 1 else (1 if m["draw"] == 1 else 2)
        rows.append(row)

    dataset = pd.DataFrame(rows)
    dataset["diff_win_rate"] = dataset["home_win_rate"] - dataset["away_win_rate"]
    dataset["diff_form"] = dataset["home_recent_form"] - dataset["away_recent_form"]

    model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model.fit(dataset[FEATURES], dataset["target"])

    joblib.dump({"model": model, "features": FEATURES}, "models/rf_v1.joblib")
    print(f"Trained on {len(dataset)} matches. Saved -> models/rf_v1.joblib")


if __name__ == "__main__":
    main()
