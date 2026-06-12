"""Daily tournament update — ONE command refreshes everything.

Usage (after filling real scores in data/results_2026.csv):
    python src/daily_update.py

What it does:
1. Recomputes Elo ratings including the real 2026 results played so far
2. Regenerates v1 predictions for UNPLAYED matches (played ones stay frozen — never edited)
3. Re-runs the 10,000-simulation Monte Carlo with played matches fixed at their real result
4. Rebuilds the projected bracket (table + image)
5. Prints the live performance report

Then: git add . && git commit && git push  ->  the dashboard updates itself.
"""

import json
import math
import sys

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.default_rng()

HEIR_NATIONS = {
    "West Germany": "Germany", "Soviet Union": "Russia", "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia", "Czechoslovakia": "Czech Republic",
    "Zaire": "DR Congo", "Dutch East Indies": "Indonesia", "German DR": "East Germany",
}
NAME_MAP = {"USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina"}
HOSTS = {"United States", "Mexico", "Canada"}
VENUE_COUNTRY = {
    "Mexico City": "Mexico", "Guadalajara (Zapopan)": "Mexico", "Monterrey (Guadalupe)": "Mexico",
    "Toronto": "Canada", "Vancouver": "Canada",
}
ELO_SIGMA = 60
N_SIMS = 10_000


# ════════════════════════════════════════════════════════════════════
# 1. ELO — replay all internationals + real 2026 results played so far
# ════════════════════════════════════════════════════════════════════

def k_factor(tournament):
    if "FIFA World Cup" in tournament and "qualification" not in tournament.lower():
        return 60
    if any(t in tournament for t in ["UEFA Euro", "Copa América", "African Cup", "AFC Asian Cup", "Gold Cup"]):
        return 50
    if "qualification" in tournament.lower() or "Nations League" in tournament:
        return 40
    return 20


def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def recompute_elo(played_results):
    intl = pd.read_csv("data/raw/historical/all_international_results.csv")
    intl["date"] = pd.to_datetime(intl["date"])
    intl = intl[intl["home_score"].notna()].copy()
    intl["home_team"] = intl["home_team"].replace(HEIR_NATIONS)
    intl["away_team"] = intl["away_team"].replace(HEIR_NATIONS)
    intl = intl.sort_values("date")

    elo = {}
    for row in intl.itertuples(index=False):
        h, a = row.home_team, row.away_team
        r_h, r_a = elo.get(h, 1500), elo.get(a, 1500)
        score = 1.0 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0.0)
        K = k_factor(row.tournament)
        exp = expected_score(r_h, r_a)
        elo[h] = r_h + K * (score - exp)
        elo[a] = r_a + K * ((1 - score) - (1 - exp))

    # Append the real 2026 World Cup results recorded so far (K=60)
    for row in played_results.sort_values("date").itertuples(index=False):
        h, a = row.home_team, row.away_team
        for t in (h, a):
            if t not in elo:
                print(f"  ⚠️ '{t}' not found in Elo history — check the name in results_2026.csv")
        r_h, r_a = elo.get(h, 1500), elo.get(a, 1500)
        score = 1.0 if row.home_score > row.away_score else (0.5 if row.home_score == row.away_score else 0.0)
        exp = expected_score(r_h, r_a)
        elo[h] = r_h + 60 * (score - exp)
        elo[a] = r_a + 60 * ((1 - score) - (1 - exp))

    ratings = pd.Series(elo, name="elo").sort_values(ascending=False).reset_index()
    ratings.columns = ["team", "elo"]
    ratings.to_csv("data/processed/elo_ratings_2026.csv", index=False)
    return elo


# ════════════════════════════════════════════════════════════════════
# 2. PREDICTIONS — refresh unplayed matches with current Elo
# ════════════════════════════════════════════════════════════════════

def build_team_matches():
    matches = pd.read_csv("data/raw/historical/matches.csv")
    m = matches[matches["tournament_name"].str.contains("Men's")].copy()
    m["year"] = m["tournament_name"].str[:4].astype(int)
    m["home_team_name"] = m["home_team_name"].replace(HEIR_NATIONS)
    m["away_team_name"] = m["away_team_name"].replace(HEIR_NATIONS)
    home = m[["year", "home_team_name", "home_team_score", "away_team_score", "home_team_win", "draw"]].copy()
    home.columns = ["year", "team", "goals_for", "goals_against", "won", "draw"]
    away = m[["year", "away_team_name", "away_team_score", "home_team_score", "away_team_win", "draw"]].copy()
    away.columns = ["year", "team", "goals_for", "goals_against", "won", "draw"]
    return pd.concat([home, away], ignore_index=True)


def team_features(team_matches, team, prefix):
    past = team_matches[team_matches["team"] == team]
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


def refresh_predictions(elo, played_keys):
    bundle = joblib.load("models/rf_v1.joblib")
    model, FEATURES = bundle["model"], bundle["features"]

    pred = pd.read_csv("data/processed/predictions_2026_group_stage_v1.csv")
    team_matches = build_team_matches()

    updated = 0
    for idx, m in pred.iterrows():
        if (m["home_team"], m["away_team"]) in played_keys:
            continue  # frozen: never edit a published prediction for a played match
        row = {}
        row.update(team_features(team_matches, m["home_team"], "home"))
        row.update(team_features(team_matches, m["away_team"], "away"))
        row["home_is_host"] = int(m["home_team"] in HOSTS)
        row["away_is_host"] = int(m["away_team"] in HOSTS)
        row["elo_home"] = elo.get(m["home_team"], 1500)
        row["elo_away"] = elo.get(m["away_team"], 1500)
        row["elo_diff"] = row["elo_home"] - row["elo_away"]
        row["diff_win_rate"] = row["home_win_rate"] - row["away_win_rate"]
        row["diff_form"] = row["home_recent_form"] - row["away_recent_form"]

        probs = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
        pred.loc[idx, "p_home_win"] = round(probs[0] * 100, 1)
        pred.loc[idx, "p_draw"] = round(probs[1] * 100, 1)
        pred.loc[idx, "p_away_win"] = round(probs[2] * 100, 1)
        pred.loc[idx, "prediction"] = ["Home win", "Draw", "Away win"][probs.argmax()]
        pred.loc[idx, "elo_home"] = row["elo_home"]
        pred.loc[idx, "elo_away"] = row["elo_away"]
        updated += 1

    pred.to_csv("data/processed/predictions_2026_group_stage_v1.csv", index=False)
    return updated


# ════════════════════════════════════════════════════════════════════
# 3. SIMULATOR — Monte Carlo with played matches fixed
# ════════════════════════════════════════════════════════════════════

def poisson_pmf(k, lam):
    return lam ** k * math.exp(-lam) / math.factorial(k)


class Tournament:
    def __init__(self, elo, results):
        with open("data/processed/poisson_params.json") as f:
            self.params = json.load(f)
        fixture = pd.read_csv("data/raw/fixture_2026/fixture_2026.csv")
        fixture["team1"] = fixture["team1"].replace(NAME_MAP)
        fixture["team2"] = fixture["team2"].replace(NAME_MAP)
        self.groups = {
            g: list(zip(df["team1"], df["team2"], df["ground"]))
            for g, df in fixture[fixture["group"].notna()].groupby("group")
        }
        ko = fixture[fixture["group"].isna()].reset_index(drop=True)
        ko["match_no"] = range(73, 73 + len(ko))
        self.ko_list = list(zip(ko["match_no"], ko["round"], ko["team1"], ko["team2"], ko["ground"]))
        self.third_slots = {ph: set(ph[1:].split("/"))
                            for ph in pd.concat([ko["team1"], ko["team2"]]) if ph.startswith("3")}
        self.teams = sorted(set(fixture[fixture["group"].notna()]["team1"]) |
                            set(fixture[fixture["group"].notna()]["team2"]))
        self.elo = elo
        # Real results: (home, away) -> (gh, ga)
        self.results = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
                        for r in results.itertuples(index=False)}
        self.sim_elo = dict(elo)

    def venue_country(self, ground):
        return VENUE_COUNTRY.get(ground, "United States")

    def lambdas(self, t1, t2, ground):
        e1 = self.sim_elo.get(t1, 1500)
        e2 = self.sim_elo.get(t2, 1500)
        country = self.venue_country(ground)
        if t1 == country and t2 != country:
            p = self.params["home_advantage"]
            d = (e1 - e2) / 400
            return (math.exp(p["home_intercept"] + p["home_coef"] * d),
                    math.exp(p["away_intercept"] + p["away_coef"] * d))
        if t2 == country and t1 != country:
            p = self.params["home_advantage"]
            d = (e2 - e1) / 400
            return (math.exp(p["away_intercept"] + p["away_coef"] * d),
                    math.exp(p["home_intercept"] + p["home_coef"] * d))
        p = self.params["neutral"]
        d = (e1 - e2) / 400
        return (math.exp(p["home_intercept"] + p["home_coef"] * d),
                math.exp(p["away_intercept"] + p["away_coef"] * d))

    def sim_match(self, t1, t2, ground):
        if (t1, t2) in self.results:  # played: fixed at real result
            return self.results[(t1, t2)]
        l1, l2 = self.lambdas(t1, t2, ground)
        return rng.poisson(l1), rng.poisson(l2)

    def ko_winner(self, t1, t2, ground):
        g1, g2 = self.sim_match(t1, t2, ground)
        if g1 != g2:
            return t1 if g1 > g2 else t2
        e1 = self.sim_elo.get(t1, 1500)
        e2 = self.sim_elo.get(t2, 1500)
        p1 = 0.5 + 0.5 * (expected_score(e1, e2) - 0.5)
        return t1 if rng.random() < p1 else t2

    def sim_group(self, g):
        stats = {}
        for t1, t2, ground in self.groups[g]:
            for t in (t1, t2):
                stats.setdefault(t, [0, 0, 0])
            g1, g2 = self.sim_match(t1, t2, ground)
            stats[t1][1] += g1 - g2; stats[t1][2] += g1
            stats[t2][1] += g2 - g1; stats[t2][2] += g2
            if g1 > g2:
                stats[t1][0] += 3
            elif g2 > g1:
                stats[t2][0] += 3
            else:
                stats[t1][0] += 1; stats[t2][0] += 1
        return sorted(stats.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[1][2], rng.random()), reverse=True)

    def assign_thirds(self, qualified):
        slot_ids = list(self.third_slots.keys())

        def backtrack(i, remaining, assignment):
            if i == len(slot_ids):
                return assignment
            for grp in list(remaining):
                if grp in self.third_slots[slot_ids[i]]:
                    res = backtrack(i + 1, remaining - {grp}, {**assignment, slot_ids[i]: grp})
                    if res:
                        return res
            return None

        res = backtrack(0, set(qualified.keys()), {})
        if res is None:
            res = dict(zip(slot_ids, qualified.keys()))
        return {sid: qualified[grp] for sid, grp in res.items()}

    def simulate_once(self):
        self.sim_elo = {t: self.elo.get(t, 1500) + rng.normal(0, ELO_SIGMA) for t in self.teams}
        reached, slots, thirds = {}, {}, {}
        for g in self.groups:
            standings = self.sim_group(g)
            letter = g[-1]
            slots[f"1{letter}"] = standings[0][0]
            slots[f"2{letter}"] = standings[1][0]
            t3, s3 = standings[2]
            thirds[letter] = (t3, s3[0], s3[1], s3[2])
            reached[standings[0][0]] = "R32"
            reached[standings[1][0]] = "R32"
        ranked = sorted(thirds.items(), key=lambda kv: (kv[1][1], kv[1][2], kv[1][3], rng.random()), reverse=True)
        qualified = {grp: d[0] for grp, d in ranked[:8]}
        for t in qualified.values():
            reached[t] = "R32"
        slots.update(self.assign_thirds(qualified))

        stage_of = {"Round of 32": "R16", "Round of 16": "QF", "Quarter-final": "SF", "Semi-final": "Final"}
        for match_no, rnd, ph1, ph2, ground in self.ko_list:
            if rnd == "Match for third place":
                continue
            t1, t2 = slots[ph1], slots[ph2]
            winner = self.ko_winner(t1, t2, ground)
            if rnd == "Final":
                reached[winner] = "Champion"
                reached[t1 if winner == t2 else t2] = "Final"
            else:
                slots[f"W{match_no}"] = winner
                reached[winner] = stage_of[rnd]
        return reached

    # ── Deterministic projection (most likely tournament) ──────────────
    def match_probs(self, t1, t2, ground):
        self.sim_elo = dict(self.elo)
        l1, l2 = self.lambdas(t1, t2, ground)
        M = np.outer([poisson_pmf(i, l1) for i in range(11)], [poisson_pmf(j, l2) for j in range(11)])
        return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()

    def project(self):
        self.sim_elo = dict(self.elo)
        slots, thirds, = {}, {}
        for g in self.groups:
            exp = {}
            for t1, t2, ground in self.groups[g]:
                for t in (t1, t2):
                    exp.setdefault(t, [0.0, 0.0])
                if (t1, t2) in self.results:  # real result: real points
                    g1, g2 = self.results[(t1, t2)]
                    exp[t1][0] += 3 if g1 > g2 else (1 if g1 == g2 else 0)
                    exp[t2][0] += 3 if g2 > g1 else (1 if g1 == g2 else 0)
                    exp[t1][1] += g1 - g2
                    exp[t2][1] += g2 - g1
                else:
                    p1, pd_, p2 = self.match_probs(t1, t2, ground)
                    l1, l2 = self.lambdas(t1, t2, ground)
                    exp[t1][0] += 3 * p1 + pd_; exp[t1][1] += l1 - l2
                    exp[t2][0] += 3 * p2 + pd_; exp[t2][1] += l2 - l1
            standings = sorted(exp.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
            letter = g[-1]
            slots[f"1{letter}"] = standings[0][0]
            slots[f"2{letter}"] = standings[1][0]
            thirds[letter] = (standings[2][0], standings[2][1][0], standings[2][1][1])
        ranked = sorted(thirds.items(), key=lambda kv: (kv[1][1], kv[1][2]), reverse=True)
        qualified = {grp: d[0] for grp, d in ranked[:8]}
        slots.update(self.assign_thirds(qualified))

        bracket = []
        for match_no, rnd, ph1, ph2, ground in self.ko_list:
            if rnd == "Match for third place":
                continue
            t1, t2 = slots[ph1], slots[ph2]
            p1, pd_, p2 = self.match_probs(t1, t2, ground)
            share = expected_score(self.elo.get(t1, 1500), self.elo.get(t2, 1500))
            win1 = p1 + pd_ * (0.5 + 0.5 * (share - 0.5))
            winner = t1 if win1 >= 0.5 else t2
            slots[f"W{match_no}"] = winner
            bracket.append({"match_no": int(match_no), "round": rnd, "team1": t1, "team2": t2,
                            "winner": winner, "win_prob": round(max(win1, 1 - win1) * 100, 1),
                            "ph1": ph1, "ph2": ph2})
        return bracket


def draw_bracket(bracket, champion):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = {b["match_no"]: b for b in bracket}
    FINAL_NO = max(info)
    DEPTH = {"Round of 32": 0, "Round of 16": 1, "Quarter-final": 2, "Semi-final": 3}
    pos, leaf_y = {}, {"L": 0, "R": 0}

    def children(m):
        return [int(p[1:]) for p in (m["ph1"], m["ph2"]) if p.startswith("W") and int(p[1:]) in info]

    def layout(no, side):
        m = info[no]
        ch = children(m)
        if not ch:
            y = leaf_y[side] * 2.0
            leaf_y[side] += 1
        else:
            y = np.mean([layout(c, side) for c in ch])
        x = DEPTH[m["round"]] if side == "L" else 7 - DEPTH[m["round"]]
        pos[no] = (x, y)
        return y

    sf_l, sf_r = children(info[FINAL_NO])
    layout(sf_l, "L"); layout(sf_r, "R")
    mid_y = (pos[sf_l][1] + pos[sf_r][1]) / 2
    pos[FINAL_NO] = (3.5, mid_y - 2.6)

    GREEN, GRAY = "#1B5E20", "#555555"
    fig, ax = plt.subplots(figsize=(17, 13))
    ax.set_xlim(-0.7, 8.2); ax.set_ylim(-1.5, 15.8); ax.axis("off")
    BW, BH = 1.45, 1.05

    def draw_links(no):
        for c in children(info[no]):
            x1, y1 = pos[c]; x2, y2 = pos[no]
            xm = (x1 + x2) / 2
            ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color="#CCCCCC", linewidth=1.0, zorder=1)
            draw_links(c)

    draw_links(FINAL_NO)
    for no, (x, y) in pos.items():
        m = info[no]
        ax.add_patch(plt.Rectangle((x - BW / 2, y - BH / 2), BW, BH, facecolor="white",
                                   edgecolor="#999999", linewidth=0.8, zorder=3))
        for i, t in enumerate([m["team1"], m["team2"]]):
            w = t == m["winner"]
            ax.text(x, y + 0.21 - i * 0.42, t, ha="center", va="center", fontsize=8.2,
                    fontweight="bold" if w else "normal", color=GREEN if w else GRAY, zorder=4)
        ax.text(x, y - 0.69, f"{m['win_prob']}%", ha="center", va="center", fontsize=6.5, color="#888888", zorder=4)

    ax.text(3.5, mid_y + 2.6, "PROJECTED CHAMPION", ha="center", fontsize=11, color="#666666", fontweight="bold")
    ax.text(3.5, mid_y + 1.7, champion.upper(), ha="center", fontsize=24, color=GREEN, fontweight="bold")
    ax.text(3.5, pos[FINAL_NO][1] + 0.95, "FINAL", ha="center", fontsize=10, color="#999999", fontweight="bold")
    ax.set_title("World Cup 2026 — Projected Bracket (updated with real results)\n"
                 "Group standings by expected points · higher win probability advances each tie",
                 fontsize=13, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig("data/processed/projected_bracket.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════

def main():
    results = pd.read_csv("data/results_2026.csv")
    played = results[results["home_score"].notna()].copy()
    print(f"📋 Real results recorded: {len(played)} matches\n")

    print("1️⃣ Recomputing Elo ratings...")
    elo = recompute_elo(played)
    top = sorted(elo.items(), key=lambda kv: -kv[1])[:5]
    print("   Top 5:", ", ".join(f"{t} ({r:.0f})" for t, r in top))

    print("\n2️⃣ Refreshing predictions for unplayed matches...")
    played_keys = set(zip(played["home_team"], played["away_team"]))
    n = refresh_predictions(elo, played_keys)
    print(f"   Updated: {n} matches (played ones stay frozen)")

    print(f"\n3️⃣ Running {N_SIMS:,} tournament simulations (played matches fixed)...")
    t = Tournament(elo, played)
    STAGE_ORDER = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "Final": 4, "Champion": 5}
    counts = {team: np.zeros(6, dtype=int) for team in t.teams}
    for _ in range(N_SIMS):
        for team, stage in t.simulate_once().items():
            counts[team][: STAGE_ORDER[stage] + 1] += 1
    odds = pd.DataFrame({team: c / N_SIMS * 100 for team, c in counts.items()},
                        index=["R32", "R16", "QF", "SF", "Final", "Champion"]).T.round(1)
    odds = odds.sort_values("Champion", ascending=False)
    odds.reset_index().rename(columns={"index": "team"}).to_csv("data/processed/tournament_odds.csv", index=False)
    print("   Top 5 champion odds:")
    print(odds.head(5)[["Champion"]].to_string())

    print("\n4️⃣ Projecting bracket...")
    bracket = t.project()
    champion = bracket[-1]["winner"]
    with open("data/processed/projected_bracket.json", "w", encoding="utf-8") as f:
        json.dump({"champion": champion, "bracket": bracket}, f, ensure_ascii=False, indent=1)
    draw_bracket(bracket, champion)
    print(f"   Projected champion: {champion}")

    print("\n5️⃣ Live performance:")
    import track_results
    track_results.main()

    print("\n✅ ALL DONE. Now: git add . && git commit -m \"Daily update\" && git push")


if __name__ == "__main__":
    main()
