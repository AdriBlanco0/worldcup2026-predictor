"""Scrape the official 2026 FIFA World Cup squads from Wikipedia.

Extracts all 26-player squads for the 48 national teams (1,248 players)
with position, date of birth, age, caps, goals and club, and saves them
as a single CSV ready for analysis.

Usage:
    python src/scrape_squads_2026.py
"""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
OUTPUT = "data/raw/squads_2026/squads_2026.csv"
HEADERS = {"User-Agent": "worldcup2026-predictor (data science portfolio project)"}


def scrape_squads():
    response = requests.get(URL, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    squads = []
    current_group = None

    # Walk the page: h2 headings carry the group, h3 headings carry the team,
    # and each team heading is followed by its squad table.
    for heading in soup.find_all(["h2", "h3"]):
        title = heading.get_text(strip=True)

        if heading.name == "h2":
            match = re.match(r"^(Group [A-L])$", title)
            current_group = match.group(1) if match else None
            continue

        # h3 inside a group section -> team name
        if current_group is None:
            continue

        team = title
        table = heading.find_next("table", class_="wikitable")
        if table is None:
            continue

        df = pd.read_html(StringIO(str(table)))[0]

        # Squad tables always have these columns; skip anything else
        if "Player" not in df.columns or "Pos." not in df.columns:
            continue

        df["team"] = team
        df["group"] = current_group
        squads.append(df)

    return pd.concat(squads, ignore_index=True)


def clean_squads(df):
    df = df.rename(columns={
        "No.": "number",
        "Pos.": "position",
        "Player": "player",
        "Date of birth (age)": "date_of_birth_raw",
        "Caps": "caps",
        "Goals": "goals",
        "Club": "club",
    })

    # Position comes as "1GK", "2DF", etc. -> keep the letters only
    df["position"] = df["position"].astype(str).str.extract(r"([A-Z]{2})")

    # Extract date of birth and age from "May 17, 2000 (aged 26)"
    dob_text = df["date_of_birth_raw"].astype(str).str.extract(r"^(.+?)\s*\(aged", expand=False)
    df["date_of_birth"] = pd.to_datetime(dob_text, format="%B %d, %Y", errors="coerce")
    df["age"] = df["date_of_birth_raw"].astype(str).str.extract(r"aged (\d+)").astype(float)

    # Captain flag comes embedded in the player name, e.g. "Lionel Messi (captain)"
    df["is_captain"] = df["player"].str.contains(r"\(c", case=False, na=False)
    df["player"] = df["player"].str.replace(r"\s*\((captain|c)\)\s*", "", regex=True).str.strip()

    cols = ["group", "team", "number", "player", "position", "date_of_birth",
            "age", "caps", "goals", "club", "is_captain"]
    return df[cols]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Scraping {URL} ...")
    raw = scrape_squads()
    print(f"Tables scraped: {raw['team'].nunique()} teams, {len(raw)} players")

    clean = clean_squads(raw)
    clean.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Saved -> {OUTPUT}")
    print(clean.head(10).to_string())
