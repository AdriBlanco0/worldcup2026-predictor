"""Pilot: scrape Big-5 European league 2025/26 player stats and match to World Cup squads.

Goal: measure how many of our 1,246 World Cup players we can enrich with current-season
club stats from FBref (the foundation for the Player Performance Tracker).
"""
import sys
import unicodedata

import pandas as pd
import soccerdata as sd

sys.stdout.reconfigure(encoding="utf-8")


def normalize(name):
    """Lowercase, strip accents — for fuzzy-ish name matching across sources."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return n.lower().strip()


print("1) Downloading Big-5 European leagues 2025/26 standard stats...")
fbref = sd.FBref(leagues="Big 5 European Leagues Combined", seasons="2025-2026")
stats = fbref.read_player_season_stats(stat_type="standard")
stats = stats.reset_index()
# Flatten multi-index columns
stats.columns = [c[1] if isinstance(c, tuple) and c[1] else (c[0] if isinstance(c, tuple) else c)
                 for c in stats.columns]
print(f"   FBref players: {len(stats)}")

stats["norm"] = stats["player"].map(normalize)
fbref_lookup = stats.set_index("norm")

print("\n2) Matching to World Cup squads...")
squads = pd.read_csv("data/raw/squads_2026/squads_2026.csv")
squads["norm"] = squads["player"].map(normalize)

matched = squads["norm"].isin(set(stats["norm"]))
squads["club_data"] = matched

print(f"   Matched: {matched.sum()} / {len(squads)} players ({matched.mean()*100:.0f}%)")
print("\n   Coverage by group:")
print(squads.groupby("group")["club_data"].mean().mul(100).round(0).to_string())

print("\n   Coverage in groups A/B/C (the completed groups):")
abc = squads[squads["group"].isin(["Group A", "Group B", "Group C"])]
print(abc.groupby("team")["club_data"].agg(["sum", "count"]).to_string())

# Save the enriched sample for matched players
cols = ["Gls", "Ast", "Min", "MP", "xG", "xAG"]
have = [c for c in cols if c in stats.columns]
enriched = squads[matched].merge(
    stats[["norm"] + have], on="norm", how="left"
)
enriched.to_csv("data/processed/pilot_player_club_stats.csv", index=False)
print(f"\n3) Saved {len(enriched)} enriched players -> data/processed/pilot_player_club_stats.csv")
print("\n   Sample (matched stars):")
sample = enriched[enriched["goals"] > 30].head(12)
show = [c for c in ["player", "team", "club", "Gls", "Ast", "Min"] if c in sample.columns]
print(sample[show].to_string(index=False))
