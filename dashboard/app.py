import json
import math

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

DATA = Path(__file__).parent.parent / "data"


def mtime(path):
    """File modification time — used as cache key so caches refresh when data files change."""
    return path.stat().st_mtime


@st.cache_data
def load_predictions(data_version):
    df = pd.read_csv(DATA / "processed" / "predictions_2026_group_stage_v1.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["kickoff_spain"] = pd.to_datetime(df["kickoff_spain"])
    return df

@st.cache_data
def load_performance(data_version):
    path = DATA / "processed" / "model_performance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_results(data_version):
    df = pd.read_csv(DATA / "results_2026.csv")
    played = df[df["home_score"].notna()]
    return {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
            for r in played.itertuples(index=False)}

@st.cache_data
def load_squads(data_version):
    return pd.read_csv(DATA / "raw" / "squads_2026" / "squads_2026.csv")

@st.cache_data
def load_poisson(data_version):
    with open(DATA / "processed" / "poisson_params.json") as f:
        params = json.load(f)
    elo = pd.read_csv(DATA / "processed" / "elo_ratings_2026.csv")
    return params, elo.set_index("team")["elo"].to_dict()

@st.cache_data
def load_odds(data_version):
    return pd.read_csv(DATA / "processed" / "tournament_odds.csv")

@st.cache_data
def load_confederations(data_version):
    path = DATA / "processed" / "confederation_stats.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_groups(data_version):
    path = DATA / "processed" / "group_table.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_thirds(data_version):
    path = DATA / "processed" / "third_place_table.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_knockout(data_version):
    path = DATA / "processed" / "knockout_predictions.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_bracket(data_version):
    with open(DATA / "processed" / "projected_bracket.json", encoding="utf-8") as f:
        return json.load(f)

pred = load_predictions(mtime(DATA / "processed" / "predictions_2026_group_stage_v1.csv"))
perf_path = DATA / "processed" / "model_performance.csv"
performance = load_performance(mtime(perf_path) if perf_path.exists() else 0)
real_results = load_results(mtime(DATA / "results_2026.csv"))
squads = load_squads(mtime(DATA / "raw" / "squads_2026" / "squads_2026.csv"))
poisson_params, current_elo = load_poisson(mtime(DATA / "processed" / "poisson_params.json"))


def poisson_pmf(k, lam):
    return lam ** k * math.exp(-lam) / math.factorial(k)


HOSTS_2026 = {"United States", "Mexico", "Canada"}


def score_matrix(home, away, elo_home, elo_away, max_goals=6):
    """Exact score probabilities. Hosts get the home-advantage model on home soil."""
    if home in HOSTS_2026 and away not in HOSTS_2026:
        p = poisson_params["home_advantage"]
        d = (elo_home - elo_away) / 400
        lam_h = math.exp(p["home_intercept"] + p["home_coef"] * d)
        lam_a = math.exp(p["away_intercept"] + p["away_coef"] * d)
    elif away in HOSTS_2026 and home not in HOSTS_2026:
        p = poisson_params["home_advantage"]
        d = (elo_away - elo_home) / 400
        lam_a = math.exp(p["home_intercept"] + p["home_coef"] * d)
        lam_h = math.exp(p["away_intercept"] + p["away_coef"] * d)
    else:
        p = poisson_params["neutral"]
        d = (elo_home - elo_away) / 400
        lam_h = math.exp(p["home_intercept"] + p["home_coef"] * d)
        lam_a = math.exp(p["away_intercept"] + p["away_coef"] * d)
    matrix = np.outer(
        [poisson_pmf(i, lam_h) for i in range(max_goals + 1)],
        [poisson_pmf(j, lam_a) for j in range(max_goals + 1)],
    )
    return matrix, lam_h, lam_a


def probability_bar(p_home, p_draw, p_away):
    """Visual stacked probability bar (green / amber / red)."""
    return f"""
    <div style="display:flex; width:100%; height:26px; border-radius:6px; overflow:hidden;
                font-size:12px; font-weight:600; color:white; text-align:center;">
      <div style="width:{p_home}%; background:#2E7D32; line-height:26px;">{p_home}%</div>
      <div style="width:{p_draw}%; background:#F9A825; line-height:26px; color:#333;">{p_draw}%</div>
      <div style="width:{p_away}%; background:#C62828; line-height:26px;">{p_away}%</div>
    </div>
    """


# ───────────────────────── HEADER ─────────────────────────
st.title("⚽ World Cup 2026 Predictor")
st.markdown(
    "**Machine Learning predictions for every match — published BEFORE each matchday, never edited.** "
    "Model v1: Random Forest + self-computed Elo ratings (49,000 internationals since 1872) · "
    "[Code on GitHub](https://github.com/AdriBlanco0/worldcup2026-predictor)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Validation accuracy (7 World Cups)", "52.4%")
c2.metric("Improvement over v0", "+3.9 pts")
c3.metric("Matches predicted", len(pred))
if performance is not None:
    live = f"{int(performance['v1_correct'].sum())}/{len(performance)}"
else:
    live = "0/0"
c4.metric("🔴 Live record (tournament)", live)

tab_pred, tab_groups, tab_bracket, tab_scores, tab_odds, tab_conf, tab_teams, tab_players, tab_model = st.tabs(
    ["🔮 Predictions", "📋 Groups", "🗺️ Bracket", "🎯 Exact Scores", "🏆 Tournament Odds",
     "🌍 Continents", "👕 Teams", "⭐ Players", "🤖 The Model"]
)


# ───────────────────────── TAB 1: PREDICTIONS ─────────────────────────
with tab_pred:
    st.subheader("⚽ Next matches")

    # Upcoming knockout ties first (the live phase)
    ko_path = DATA / "processed" / "knockout_predictions.csv"
    ko = load_knockout(mtime(ko_path) if ko_path.exists() else 0)
    if ko is not None and len(ko):
        upcoming = ko[ko["home_score"].isna()].copy()
        if "kickoff_spain" in upcoming:
            upcoming = upcoming.sort_values("kickoff_spain")
        if len(upcoming):
            st.caption("Knockout stage — probability of advancing (extra time & penalties included). "
                       "Kickoff in Spanish time 🇪🇸")
            for _, m in upcoming.head(8).iterrows():
                with st.container(border=True):
                    left, right = st.columns([3, 2])
                    ko_time = ""
                    if pd.notna(m.get("kickoff_spain")):
                        ko_time = pd.to_datetime(m["kickoff_spain"]).strftime("%d %b · %H:%M 🇪🇸")
                    left.markdown(f"### {m['team1']} 🆚 {m['team2']}")
                    left.caption(f"{m['round']} · {ko_time}")
                    right.markdown(f"**{m['favorite']} advances ({max(m['p_team1_adv'], m['p_team2_adv']):.0f}%)** · pred {m['pred_score']}")
                    right.markdown(probability_bar(round(m["p_team1_adv"], 0), 0, round(m["p_team2_adv"], 0)),
                                   unsafe_allow_html=True)
            st.caption("Full bracket and results in the **🗺️ Bracket** tab.")
        else:
            st.success("All knockout ties so far are played — see the **🗺️ Bracket** tab.", icon="🏆")

    st.divider()
    with st.expander("📚 Group stage — predictions & results (completed)"):
        st.caption(
            "**Model: Random Forest v1** (World Cup history 1962-2022 + current Elo ratings) · "
            "🟢 home win · 🟡 draw · 🔴 away win — kickoff times in Spanish time (CEST)"
        )

        dates = sorted(pred["date"].dt.date.unique())

        # Default to the first date that still has unplayed matches; if none (group stage over),
        # show the last matchday rather than day 1.
        def date_has_pending(d):
            day = pred[pred["date"].dt.date == d]
            return any((m["home_team"], m["away_team"]) not in real_results for _, m in day.iterrows())

        pending_idx = next((i for i, d in enumerate(dates) if date_has_pending(d)), None)
        if pending_idx is None:
            st.success("✅ Group stage complete — knockout predictions are in the **🗺️ Bracket** tab.", icon="🏆")
            default_idx = len(dates) - 1
        else:
            default_idx = pending_idx
        selected_date = st.selectbox("Pick a group-stage date", dates, index=default_idx)
        day_matches = pred[pred["date"].dt.date == selected_date].sort_values("kickoff_spain")

        for _, m in day_matches.iterrows():
            key = (m["home_team"], m["away_team"])
            with st.container(border=True):
                left, right = st.columns([2, 3])
                left.markdown(f"### {m['home_team']} 🆚 {m['away_team']}")
                left.caption(f"Group {m['group'][-1]} · {m['kickoff_spain'].strftime('%d %b · %H:%M')} 🇪🇸")
                exact = ""
                if "pred_home_goals" in m and pd.notna(m["pred_home_goals"]):
                    exact = f"{int(m['pred_home_goals'])}-{int(m['pred_away_goals'])}"
                if key in real_results:
                    hs, as_ = real_results[key]
                    actual = "Home win" if hs > as_ else ("Away win" if as_ > hs else "Draw")
                    verdict = "✅ Model was right" if actual == m["prediction"] else "❌ Model was wrong"
                    left.markdown(f"## ⚽ FINAL: {hs} - {as_}")
                    right.markdown(f"**Model pick (frozen pre-match): {m['prediction']}** → {verdict}")
                    if exact:
                        exact_hit = "🎯 EXACT SCORE NAILED!" if exact == f"{hs}-{as_}" else ""
                        right.caption(f"Predicted exact score: {exact}  {exact_hit}")
                else:
                    right.markdown(f"**Model pick: {m['prediction']}**")
                    if exact:
                        right.caption(f"🎯 Predicted exact score: {exact}")
                right.markdown(probability_bar(m["p_home_win"], m["p_draw"], m["p_away_win"]),
                               unsafe_allow_html=True)

        st.divider()
        st.subheader("All group-stage predictions")

        groups = ["All"] + sorted(pred["group"].unique())
        selected_group = st.selectbox("Filter by group", groups)
        table = pred if selected_group == "All" else pred[pred["group"] == selected_group]

        st.dataframe(
            table[["kickoff_spain", "group", "home_team", "away_team",
                   "p_home_win", "p_draw", "p_away_win", "prediction"]],
            use_container_width=True, hide_index=True,
            column_config={
                "kickoff_spain": st.column_config.DatetimeColumn("Kickoff 🇪🇸", format="DD MMM · HH:mm"),
                "group": "Group",
                "home_team": "Home",
                "away_team": "Away",
                "p_home_win": st.column_config.ProgressColumn("Home %", min_value=0, max_value=100, format="%.1f%%"),
                "p_draw": st.column_config.ProgressColumn("Draw %", min_value=0, max_value=100, format="%.1f%%"),
                "p_away_win": st.column_config.ProgressColumn("Away %", min_value=0, max_value=100, format="%.1f%%"),
                "prediction": "Model pick",
            },
        )


# ───────────────────────── TAB: GROUPS ─────────────────────────
with tab_groups:
    st.subheader("📋 Group standings & qualification odds")
    st.caption("Live standings from real results + probability of advancing to the Round of 32 "
               "(top 2 of each group + the 8 best third-placed teams), from 10,000 Monte Carlo simulations.")

    groups = load_groups(mtime(DATA / "processed" / "group_table.csv")
                         if (DATA / "processed" / "group_table.csv").exists() else 0)
    if groups is None or len(groups) == 0:
        st.info("Group tables will appear once matches are played.")
    else:
        group_names = sorted(groups["group"].unique())
        cols = st.columns(2)
        for i, gname in enumerate(group_names):
            gdf = groups[groups["group"] == gname].copy()
            with cols[i % 2]:
                st.markdown(f"### {gname}")
                show_cols = ["team", "played", "gd", "points", "p_advance"]
                if "status" in gdf.columns:
                    show_cols.append("status")
                st.dataframe(
                    gdf[show_cols],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "team": "Team", "played": "PJ", "gd": "GD", "points": "Pts",
                        "p_advance": st.column_config.ProgressColumn(
                            "Advance %", min_value=0, max_value=100, format="%.0f%%"),
                        "status": "Status",
                    },
                )
        st.caption("**Status:** ✅ Qualified = advances in every one of the 10,000 simulations · "
                   "❌ Eliminated = advances in none of them · 🪙 = only a best-third path left · ⚪ Alive. "
                   "Standings & qualification use the **2026 head-to-head-first tiebreaker** (a rule change "
                   "from previous World Cups, where goal difference came first).")

        st.divider()
        st.subheader("🪙 Best third-placed teams")
        st.caption("The 8 best third-placed teams (across the 12 groups) also reach the Round of 32. "
                   "Ranked by points → goal difference → goals scored.")
        thirds = load_thirds(mtime(DATA / "processed" / "third_place_table.csv")
                             if (DATA / "processed" / "third_place_table.csv").exists() else 0)
        if thirds is not None and len(thirds) > 0:
            tshow = thirds.copy()
            tshow["zone"] = np.where(tshow["qualifies"], "✅ In", "❌ Out")
            st.dataframe(
                tshow[["rank", "team", "group", "points", "gd", "gf", "zone"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "rank": "#", "team": "Team", "group": "Group",
                    "points": "Pts", "gd": "GD", "gf": "GF", "zone": "Top 8?",
                },
            )


# ───────────────────────── TAB: BRACKET ─────────────────────────
with tab_bracket:
    st.subheader("🗺️ Knockout bracket")
    st.caption(
        "The real Round-of-32 matchups and, per tie, the model's probability of each team advancing "
        "(extra time & penalties included). The bracket image shows the single most likely path to the trophy."
    )

    bimg = DATA / "processed" / "knockout_bracket.png"
    if bimg.exists():
        st.image(str(bimg), use_column_width=True)
        st.caption("Bold/green = team the model projects to advance. Real results override projections as ties are played.")
        st.divider()

    # Real knockout ties with predictions (once the knockout stage has started)
    ko_path = DATA / "processed" / "knockout_predictions.csv"
    ko = load_knockout(mtime(ko_path) if ko_path.exists() else 0)
    if ko is not None and len(ko):
        for rnd in ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]:
            rmatches = ko[ko["round"] == rnd]
            if len(rmatches) == 0:
                continue
            st.markdown(f"### {rnd}")
            for _, m in rmatches.iterrows():
                fav_p = max(m["p_team1_adv"], m["p_team2_adv"])
                played = pd.notna(m.get("home_score"))
                with st.container(border=True):
                    left, right = st.columns([3, 2])
                    left.markdown(f"**{m['team1']} 🆚 {m['team2']}**")
                    ko_time = ""
                    if "kickoff_spain" in m and pd.notna(m.get("kickoff_spain")):
                        ko_time = pd.to_datetime(m["kickoff_spain"]).strftime("%d %b · %H:%M 🇪🇸 · ")
                    left.caption(f"{ko_time}Model: {m['favorite']} advances ({fav_p:.0f}%) · predicted {m['pred_score']}")
                    if played:
                        hs, as_ = int(m["home_score"]), int(m["away_score"])
                        adv = m.get("advanced")
                        ok = "✅" if adv == m["favorite"] else "❌"
                        pens = " (pens)" if hs == as_ else ""
                        right.markdown(f"### {hs} - {as_} {ok}")
                        right.caption(f"{adv} advanced{pens}")
                    else:
                        right.markdown(probability_bar(round(m["p_team1_adv"], 0), 0, round(m["p_team2_adv"], 0)),
                                       unsafe_allow_html=True)
    else:
        st.info("Knockout predictions will appear when the group stage finishes.")


# ───────────────────────── TAB 2: EXACT SCORES ─────────────────────────
with tab_scores:
    st.subheader("Exact score probabilities — Poisson model")
    st.caption(
        "**Model: Poisson regression** — a DIFFERENT model from the Predictions tab. "
        "Each team's expected goals are estimated from the current Elo gap (trained on 32,000+ "
        "internationals since 1990), then Poisson gives the probability of every exact score."
    )
    st.warning(
        "⚠️ The probabilities here may differ from the Predictions tab — they come from two "
        "independent models. The Poisson model uses up-to-date Elo ratings (current form) and "
        "carries real home advantage from its training data; the Random Forest only knows "
        "World Cup history. Disagreement between models is normal — and interesting.",
        icon="🤖",
    )

    match_options = pred.sort_values("kickoff_spain").apply(
        lambda r: f"{r['home_team']} vs {r['away_team']} ({r['kickoff_spain'].strftime('%d %b')})", axis=1
    )
    selected_match = st.selectbox("Pick a match", match_options)

    sel = pred.sort_values("kickoff_spain").iloc[list(match_options).index(selected_match)]
    home, away = sel["home_team"], sel["away_team"]
    elo_h = current_elo.get(home, 1500)
    elo_a = current_elo.get(away, 1500)

    matrix, lam_h, lam_a = score_matrix(home, away, elo_h, elo_a)
    n = matrix.shape[0]
    if home in HOSTS_2026 or away in HOSTS_2026:
        st.caption("🏟️ Host playing on home soil — home-advantage model applied.")

    c1, c2 = st.columns([3, 2])

    with c1:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(matrix * 100, cmap="Greens")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xlabel(f"{away} goals"); ax.set_ylabel(f"{home} goals")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]*100:.1f}", ha="center", va="center",
                        color="white" if matrix[i, j] > 0.06 else "black", fontsize=8)
        ax.set_title(f"{home} (Elo {elo_h:.0f}) vs {away} (Elo {elo_a:.0f})", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.metric("Expected goals", f"{lam_h:.2f} — {lam_a:.2f}")

        p_home = np.tril(matrix, -1).sum() * 100
        p_draw = np.trace(matrix) * 100
        p_away = np.triu(matrix, 1).sum() * 100
        st.markdown(probability_bar(round(p_home, 1), round(p_draw, 1), round(p_away, 1)),
                    unsafe_allow_html=True)
        st.caption(f"🟢 {home} {p_home:.1f}% · 🟡 draw {p_draw:.1f}% · 🔴 {away} {p_away:.1f}%")

        flat = [(f"{i}-{j}", matrix[i, j]) for i in range(n) for j in range(n)]
        top5 = sorted(flat, key=lambda t: -t[1])[:5]
        st.markdown("**Most likely scores:**")
        for s, p in top5:
            st.markdown(f"- **{s}** — {p*100:.1f}%")

    st.info(
        "💡 Note: even the most likely exact score rarely exceeds ~14% — football is beautifully "
        "unpredictable. The value is in the full distribution, not a single guess."
    )


# ───────────────────────── TAB 3: TOURNAMENT ODDS ─────────────────────────
with tab_odds:
    st.subheader("Who wins the World Cup?")

    ko_odds_path = DATA / "processed" / "knockout_odds.csv"
    knockout_phase = ko_odds_path.exists()

    if knockout_phase:
        st.caption("**10,000 Monte Carlo simulations of the REAL knockout bracket** — every tie "
                   "simulated with the Poisson + Elo model (extra time & penalties included), winners "
                   "advanced through the actual bracket. Re-run after each match.")
        odds = load_odds(mtime(ko_odds_path)).sort_values("Champion", ascending=False)
        stage_cols = ["R16", "QF", "SF", "Final", "Champion"]
        img = DATA / "processed" / "knockout_champion_odds.png"
    else:
        st.caption("**10,000 Monte Carlo simulations** of the full tournament. Re-computed after every matchday.")
        odds = load_odds(mtime(DATA / "processed" / "tournament_odds.csv")).sort_values("Champion", ascending=False)
        stage_cols = ["R32", "R16", "QF", "SF", "Final", "Champion"]
        img = None

    if img and img.exists():
        st.image(str(img), use_column_width=True)
    else:
        top15 = odds.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(top15["team"], top15["Champion"], color="#2E7D32")
        for bar, val in zip(bars, top15["Champion"]):
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("P(Champion) %")
        ax.set_title("Champion probability — top 15", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Full table — probability of reaching each stage (%):**")
    labels = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-final",
              "SF": "Semi-final", "Final": "Final", "Champion": "🏆 Champion"}
    cfg = {"team": "Team"}
    for c in stage_cols:
        cfg[c] = st.column_config.ProgressColumn(labels[c], min_value=0, max_value=100, format="%.1f%%")
    st.dataframe(odds[["team"] + stage_cols], use_container_width=True, hide_index=True, height=600,
                 column_config=cfg)

    st.info(
        "💡 Note: this model is more bullish on the favourites than betting markets "
        "(its Poisson slope is steep and knockout shootouts are Elo-weighted). The ranking "
        "matches the consensus; the magnitudes are the model's own opinion.",
        icon="📊",
    )


# ───────────────────────── TAB: CONTINENTS ─────────────────────────
with tab_conf:
    st.subheader("🌍 Which continent is winning the World Cup?")
    st.caption("Performance by confederation, computed with SQL on the project's MySQL database. "
               "Points = 3 win / 1 draw / 0 loss, averaged per team-appearance.")

    conf = load_confederations(mtime(DATA / "processed" / "confederation_stats.csv")
                               if (DATA / "processed" / "confederation_stats.csv").exists() else 0)
    if conf is None or len(conf) == 0:
        st.info("Confederation stats will appear once matches have been played.")
    else:
        names = {"UEFA": "🇪🇺 UEFA (Europe)", "CONMEBOL": "🌎 CONMEBOL (S. America)",
                 "CAF": "🌍 CAF (Africa)", "AFC": "🌏 AFC (Asia)",
                 "CONCACAF": "🌎 CONCACAF (N. America)", "OFC": "🇳🇿 OFC (Oceania)"}
        conf = conf.copy()
        conf["region"] = conf["confederation"].map(names).fillna(conf["confederation"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Average points per match**")
            fig, ax = plt.subplots(figsize=(6, 4))
            d = conf.sort_values("avg_points")
            ax.barh(d["confederation"], d["avg_points"], color="#1565C0")
            for i, v in enumerate(d["avg_points"]):
                ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
            ax.set_xlabel("Avg points (3=win, 1=draw)")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig); plt.close(fig)
        with c2:
            st.markdown("**Combined title probability**")
            fig, ax = plt.subplots(figsize=(6, 4))
            d = conf.sort_values("champion_pct")
            ax.barh(d["confederation"], d["champion_pct"], color="#2E7D32")
            for i, v in enumerate(d["champion_pct"]):
                ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
            ax.set_xlabel("Sum of champion probability (%)")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig); plt.close(fig)

        st.markdown("**Full breakdown**")
        st.dataframe(
            conf[["region", "matches_played", "wins", "draws", "losses",
                  "avg_points", "avg_goals_for", "avg_goals_against", "champion_pct"]],
            use_container_width=True, hide_index=True,
            column_config={
                "region": "Confederation", "matches_played": "Played",
                "wins": "W", "draws": "D", "losses": "L",
                "avg_points": "Avg pts", "avg_goals_for": "Avg GF",
                "avg_goals_against": "Avg GA",
                "champion_pct": st.column_config.ProgressColumn("🏆 Title %", min_value=0, max_value=100, format="%.1f%%"),
            },
        )


# ───────────────────────── TAB 4: TEAMS ─────────────────────────
with tab_teams:
    st.subheader("Team explorer")

    team = st.selectbox("Pick a team", sorted(squads["team"].unique()))
    team_squad = squads[squads["team"] == team]

    captain = team_squad.loc[team_squad["is_captain"], "player"]
    captain_name = captain.iloc[0] if len(captain) else "—"
    top_scorer = team_squad.sort_values("goals", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Squad size", len(team_squad))
    c2.metric("Average age", f"{team_squad['age'].mean():.1f}")
    c3.metric("Captain", captain_name)
    c4.metric("Top scorer", f"{top_scorer['player']} ({top_scorer['goals']})")

    st.markdown(f"**{team}'s group-stage matches:**")
    team_games = pred[(pred["home_team"] == team) | (pred["away_team"] == team)].sort_values("kickoff_spain")
    for _, m in team_games.iterrows():
        with st.container(border=True):
            left, right = st.columns([2, 3])
            left.markdown(f"**{m['home_team']} 🆚 {m['away_team']}**")
            left.caption(f"{m['kickoff_spain'].strftime('%d %b · %H:%M')} 🇪🇸")
            right.markdown(probability_bar(m["p_home_win"], m["p_draw"], m["p_away_win"]),
                           unsafe_allow_html=True)

    st.markdown("**Full squad:**")
    st.dataframe(
        team_squad[["number", "player", "position", "age", "caps", "goals", "club"]]
        .sort_values("number"),
        use_container_width=True, hide_index=True,
        column_config={
            "number": "#", "player": "Player", "position": "Pos",
            "age": "Age", "caps": "Caps", "goals": "Goals", "club": "Club",
        },
    )


# ───────────────────────── TAB: PLAYERS ─────────────────────────
with tab_players:
    st.subheader("⭐ The stars of the World Cup")
    st.caption("Player profiles from the official squads (caps & international goals before the "
               "tournament). The foundation for the upcoming live Performance Tracker — "
               "expected vs actual output during the tournament.")

    sq = squads.copy()
    sq["goals_per_cap"] = (sq["goals"] / sq["caps"]).round(2)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🥅 Top international scorers**")
        st.dataframe(
            sq.nlargest(15, "goals")[["player", "team", "goals", "caps", "age"]],
            use_container_width=True, hide_index=True,
            column_config={"player": "Player", "team": "Team", "goals": "Goals",
                           "caps": "Caps", "age": "Age"},
        )
    with c2:
        st.markdown("**🎯 Most lethal (goals per cap, min. 40 caps)**")
        st.dataframe(
            sq[sq["caps"] >= 40].nlargest(15, "goals_per_cap")[["player", "team", "goals_per_cap", "goals", "caps"]],
            use_container_width=True, hide_index=True,
            column_config={"player": "Player", "team": "Team", "goals_per_cap": "G/cap",
                           "goals": "Goals", "caps": "Caps"},
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**🎖️ Most experienced (caps)**")
        st.dataframe(
            sq.nlargest(15, "caps")[["player", "team", "caps", "goals", "age"]],
            use_container_width=True, hide_index=True,
            column_config={"player": "Player", "team": "Team", "caps": "Caps",
                           "goals": "Goals", "age": "Age"},
        )
    with c4:
        st.markdown("**🌱 Rising stars (21 or younger, by caps)**")
        st.dataframe(
            sq[sq["age"] <= 21].nlargest(15, "caps")[["player", "team", "age", "caps", "goals"]],
            use_container_width=True, hide_index=True,
            column_config={"player": "Player", "team": "Team", "age": "Age",
                           "caps": "Caps", "goals": "Goals"},
        )

    st.info("🔜 **Coming next:** the live Performance Tracker — once tournament goals are logged, this "
            "tab will show who is over- and under-performing their expected output (the revelations "
            "and disappointments of the World Cup).")


# ───────────────────────── TAB 3: THE MODEL ─────────────────────────
with tab_model:
    st.subheader("How the model works")
    st.markdown("""
    | | |
    |---|---|
    | **Algorithm** | Random Forest (300 trees, max depth 6) |
    | **Training data** | All men's World Cup group-stage matches 1962-2022 (636 matches) |
    | **Features (v1)** | Elo ratings (self-computed over 49,000 internationals since 1872, K by tournament importance), historical win rate, goals for/against, World Cup experience, recent form, host advantage |
    | **Evaluation** | Leave-one-tournament-out: tested separately on each World Cup 1998-2022 |
    | **Validation accuracy** | **52.4%** vs 48.5% (v0 without Elo) — v1 wins in 5 of 7 tournaments, never loses |

    **No data leakage:** every match's features are computed using only matches played *before* that
    tournament — the model never sees the future.

    **Known limitations:**
    - Draws remain the hardest class (~25% of matches, weakly predicted)
    - 2026 has 3 hosts for the first time — host advantage may be diluted
    - Defunct nations mapped to FIFA heirs (West Germany → Germany, Soviet Union → Russia...)

    **Coming in v2:** squad features (age, caps, tournament experience), exact-score calibration
    and the Player Performance Tracker.
    """)

    st.subheader("📏 Is the model well calibrated?")
    cal_path = DATA / "processed" / "calibration.png"
    if cal_path.exists():
        cc1, cc2 = st.columns([3, 2])
        cc1.image(str(cal_path), use_column_width=True)
        cc2.markdown("""
        A **reliability diagram**: when the model says *X%*, does it actually happen *X%* of the time?
        Points on the diagonal = honest probabilities.

        **Expected Calibration Error (ECE) = 0.031** — well below the 0.05 "well-calibrated" threshold.

        This means the percentages here aren't just rankings: a **70% really means ~7 times out of 10**.
        That trustworthiness is what makes the Monte Carlo title odds meaningful. See
        [notebook 08](https://github.com/AdriBlanco0/worldcup2026-predictor/blob/main/notebooks/08_calibration.ipynb).
        """)
    st.divider()

    st.subheader("📈 Live performance during the tournament")
    if performance is None or len(performance) == 0:
        st.info("After every matchday, real results and model accuracy are published here — hits and misses alike.")
    else:
        n = len(performance)
        v1_ok = int(performance["v1_correct"].sum())
        v0_ok = int(performance["v0_correct"].sum())
        has_exact = "exact_correct" in performance.columns
        ex_ok = int(performance["exact_correct"].sum()) if has_exact else 0
        has_rps = "v1_rps" in performance.columns
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Matches played", n)
        m2.metric("v1 outcome", f"{v1_ok}/{n} ({v1_ok/n*100:.0f}%)")
        m3.metric("v0 outcome", f"{v0_ok}/{n} ({v0_ok/n*100:.0f}%)")
        m4.metric("🎯 Exact scores", f"{ex_ok}/{n} ({ex_ok/n*100:.0f}%)")
        if has_rps:
            m5.metric("📐 RPS (v1)", f"{performance['v1_rps'].mean():.3f}",
                      help="Ranked Probability Score — the proper metric for ordered probabilistic "
                           "forecasts (lower is better). Historical model ≈ 0.20. Accuracy alone is "
                           "misleading for a probabilistic model.")

    if performance is not None and len(performance) > 0:
        st.caption("📐 **Why RPS?** Accuracy only checks if the single most-likely pick happened. "
                   "The Ranked Probability Score scores the whole probability distribution — rewarding "
                   "the model for putting honest weight on draws even when they aren't the top pick. "
                   "By RPS the model performs close to its historical level, despite a noisy accuracy in a small sample.")

        show = performance.copy()
        show["score"] = show["home_score"].astype(int).astype(str) + "-" + show["away_score"].astype(int).astype(str)
        show["v1"] = np.where(show["v1_correct"], "✅", "❌")
        if has_exact:
            show["pred"] = (show["pred_home_goals"].astype("Int64").astype(str) + "-"
                            + show["pred_away_goals"].astype("Int64").astype(str))
            show["🎯"] = np.where(show["exact_correct"], "🎯", "")
            cols = ["date", "home_team", "score", "away_team", "actual", "v1_pick", "v1", "pred", "🎯"]
            cfg = {"date": "Date", "home_team": "Home", "score": "Score", "away_team": "Away",
                   "actual": "Result", "v1_pick": "v1 pick", "v1": "v1",
                   "pred": "Exact pred", "🎯": "Hit"}
        else:
            cols = ["date", "home_team", "score", "away_team", "actual", "v1_pick", "v1"]
            cfg = {"date": "Date", "home_team": "Home", "score": "Score", "away_team": "Away",
                   "actual": "Result", "v1_pick": "v1 pick", "v1": "v1"}
        st.dataframe(show[cols], use_container_width=True, hide_index=True, column_config=cfg)
        st.caption("Both model versions are tracked head-to-head for full transparency — "
                   "matchday 1 was predicted with v0 (published before the tournament); v1 takes over from matchday 2.")

st.divider()
st.caption("Built by Adrián Blanco · [LinkedIn](https://www.linkedin.com/in/adrianblancoajenjo/) · [GitHub](https://github.com/AdriBlanco0/worldcup2026-predictor) · Predictions are published before each matchday and never edited.")
