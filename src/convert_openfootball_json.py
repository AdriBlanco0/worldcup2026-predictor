"""Convert openfootball World Cup JSON files to flat CSVs.

Converts the 2022 results and the 2026 fixture into match-per-row CSVs
consistent with the historical dataset.

Usage:
    python src/convert_openfootball_json.py
"""

import json

import pandas as pd

FILES = [
    ("data/raw/historical/worldcup_2022.json", "data/raw/historical/matches_2022.csv"),
    ("data/raw/fixture_2026/worldcup_2026_fixture.json", "data/raw/fixture_2026/fixture_2026.csv"),
]


def json_to_df(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for match in data["matches"]:
        rows.append({
            "round": match.get("round"),
            "group": match.get("group"),
            "date": match.get("date"),
            "time": match.get("time"),
            "team1": match.get("team1"),
            "team2": match.get("team2"),
            "score1": (match.get("score") or {}).get("ft", [None, None])[0],
            "score2": (match.get("score") or {}).get("ft", [None, None])[1],
            "ground": match.get("ground"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for src, dst in FILES:
        df = json_to_df(src)
        df.to_csv(dst, index=False, encoding="utf-8")
        print(f"{src} -> {dst} ({len(df)} matches)")
