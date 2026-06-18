"""Build the worldcup2026 MySQL database from the project CSVs.

Creates a normalized relational schema (teams, matches, predictions, odds, players)
with primary and foreign keys, then loads every CSV into it.

Usage:
    python src/build_database.py
"""

import os
import sys

import pandas as pd
import mysql.connector

sys.stdout.reconfigure(encoding="utf-8")

# Local dev credentials — password read from env var (set MYSQL_PASSWORD), with a local fallback.
DB_CONFIG = {"host": "localhost", "user": "root",
             "password": os.environ.get("MYSQL_PASSWORD", "mundial2026")}
DB_NAME = "worldcup2026"

# Confederation of each of the 48 qualified teams
CONFEDERATION = {
    # UEFA (Europe)
    "Austria": "UEFA", "Belgium": "UEFA", "Bosnia and Herzegovina": "UEFA", "Croatia": "UEFA",
    "Czech Republic": "UEFA", "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Netherlands": "UEFA", "Norway": "UEFA", "Portugal": "UEFA", "Scotland": "UEFA",
    "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA", "Turkey": "UEFA",
    # CONMEBOL (South America)
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Uruguay": "CONMEBOL",
    # CAF (Africa)
    "Algeria": "CAF", "Cape Verde": "CAF", "DR Congo": "CAF", "Egypt": "CAF", "Ghana": "CAF",
    "Ivory Coast": "CAF", "Morocco": "CAF", "Senegal": "CAF", "South Africa": "CAF", "Tunisia": "CAF",
    # AFC (Asia)
    "Australia": "AFC", "Iran": "AFC", "Iraq": "AFC", "Japan": "AFC", "Jordan": "AFC",
    "Qatar": "AFC", "Saudi Arabia": "AFC", "South Korea": "AFC", "Uzbekistan": "AFC",
    # CONCACAF (North & Central America)
    "Canada": "CONCACAF", "Curaçao": "CONCACAF", "Haiti": "CONCACAF", "Mexico": "CONCACAF",
    "Panama": "CONCACAF", "United States": "CONCACAF",
    # OFC (Oceania)
    "New Zealand": "OFC",
}

SCHEMA = """
CREATE TABLE teams (
    team_id        INT AUTO_INCREMENT PRIMARY KEY,
    team_name      VARCHAR(50) UNIQUE NOT NULL,
    confederation  VARCHAR(10) NOT NULL,
    elo_rating     DECIMAL(6,1)
);

CREATE TABLE matches (
    match_id       INT AUTO_INCREMENT PRIMARY KEY,
    match_date     DATE,
    group_name     VARCHAR(10),
    home_team_id   INT NOT NULL,
    away_team_id   INT NOT NULL,
    venue          VARCHAR(60),
    kickoff_spain  DATETIME,
    home_score     INT,
    away_score     INT,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE TABLE predictions (
    pred_id            INT AUTO_INCREMENT PRIMARY KEY,
    match_id           INT NOT NULL,
    p_home_win         DECIMAL(5,1),
    p_draw             DECIMAL(5,1),
    p_away_win         DECIMAL(5,1),
    predicted_outcome  VARCHAR(10),
    pred_home_goals    INT,
    pred_away_goals    INT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE odds (
    team_id        INT PRIMARY KEY,
    round_of_32    DECIMAL(5,1),
    round_of_16    DECIMAL(5,1),
    quarter_final  DECIMAL(5,1),
    semi_final     DECIMAL(5,1),
    final_p        DECIMAL(5,1),
    champion       DECIMAL(5,1),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE players (
    player_id    INT AUTO_INCREMENT PRIMARY KEY,
    team_id      INT NOT NULL,
    player_name  VARCHAR(100),
    position     VARCHAR(5),
    age          INT,
    caps         INT,
    goals        INT,
    club         VARCHAR(100),
    is_captain   BOOLEAN,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
"""


def main():
    # ── Create database fresh ────────────────────────────────────────
    cn = mysql.connector.connect(**DB_CONFIG)
    cur = cn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    cur.execute(f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4")
    cur.execute(f"USE {DB_NAME}")
    for stmt in SCHEMA.strip().split(";\n"):
        if stmt.strip():
            cur.execute(stmt)
    print(f"✅ Database '{DB_NAME}' created with 5 tables")

    # ── Load data ────────────────────────────────────────────────────
    squads = pd.read_csv("data/raw/squads_2026/squads_2026.csv")
    elo = pd.read_csv("data/processed/elo_ratings_2026.csv").set_index("team")["elo"].to_dict()
    preds = pd.read_csv("data/processed/predictions_2026_group_stage_v1.csv")
    results = pd.read_csv("data/results_2026.csv")
    odds = pd.read_csv("data/processed/tournament_odds.csv")

    # teams
    teams = sorted(squads["team"].unique())
    team_id = {}
    for t in teams:
        cur.execute("INSERT INTO teams (team_name, confederation, elo_rating) VALUES (%s,%s,%s)",
                    (t, CONFEDERATION[t], round(elo.get(t, 1500), 1)))
        team_id[t] = cur.lastrowid
    print(f"✅ teams: {len(teams)} rows")

    # matches (group stage) — join predictions with real results
    res_map = {(r.home_team, r.away_team): (r.home_score, r.away_score)
               for r in results.itertuples(index=False)}
    match_id = {}
    for r in preds.itertuples(index=False):
        hs, as_ = res_map.get((r.home_team, r.away_team), (None, None))
        hs = int(hs) if pd.notna(hs) else None
        as_ = int(as_) if pd.notna(as_) else None
        cur.execute(
            "INSERT INTO matches (match_date, group_name, home_team_id, away_team_id, venue, kickoff_spain, home_score, away_score) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (r.date, r.group, team_id[r.home_team], team_id[r.away_team],
             getattr(r, "ground", None) if hasattr(r, "ground") else None,
             r.kickoff_spain if pd.notna(r.kickoff_spain) else None, hs, as_))
        match_id[(r.home_team, r.away_team)] = cur.lastrowid
    print(f"✅ matches: {len(preds)} rows")

    # predictions
    for r in preds.itertuples(index=False):
        cur.execute(
            "INSERT INTO predictions (match_id, p_home_win, p_draw, p_away_win, predicted_outcome, pred_home_goals, pred_away_goals) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (match_id[(r.home_team, r.away_team)], r.p_home_win, r.p_draw, r.p_away_win,
             r.prediction,
             int(r.pred_home_goals) if pd.notna(r.pred_home_goals) else None,
             int(r.pred_away_goals) if pd.notna(r.pred_away_goals) else None))
    print(f"✅ predictions: {len(preds)} rows")

    # odds
    for r in odds.itertuples(index=False):
        if r.team not in team_id:
            continue
        cur.execute(
            "INSERT INTO odds (team_id, round_of_32, round_of_16, quarter_final, semi_final, final_p, champion) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (team_id[r.team], r.R32, r.R16, r.QF, r.SF, r.Final, r.Champion))
    print(f"✅ odds: {len(odds)} rows")

    # players
    n = 0
    for r in squads.itertuples(index=False):
        cur.execute(
            "INSERT INTO players (team_id, player_name, position, age, caps, goals, club, is_captain) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (team_id[r.team], r.player, r.position,
             int(r.age) if pd.notna(r.age) else None,
             int(r.caps) if pd.notna(r.caps) else None,
             int(r.goals) if pd.notna(r.goals) else None,
             r.club, bool(r.is_captain)))
        n += 1
    print(f"✅ players: {n} rows")

    cn.commit()
    cur.close()
    cn.close()
    print(f"\n🗄️  Database '{DB_NAME}' built and loaded successfully.")


if __name__ == "__main__":
    main()
