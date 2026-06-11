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

@st.cache_data
def load_predictions():
    df = pd.read_csv(DATA / "processed" / "predictions_2026_group_stage_v0.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["kickoff_spain"] = pd.to_datetime(df["kickoff_spain"])
    return df

@st.cache_data
def load_squads():
    return pd.read_csv(DATA / "raw" / "squads_2026" / "squads_2026.csv")

@st.cache_data
def load_poisson():
    with open(DATA / "processed" / "poisson_params.json") as f:
        params = json.load(f)
    elo = pd.read_csv(DATA / "processed" / "elo_ratings_2026.csv")
    return params, elo.set_index("team")["elo"].to_dict()

@st.cache_data
def load_odds():
    return pd.read_csv(DATA / "processed" / "tournament_odds.csv")

pred = load_predictions()
squads = load_squads()
poisson_params, current_elo = load_poisson()


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
    "Model v0: Random Forest trained on World Cups 1962-2022 · "
    "[Code on GitHub](https://github.com/AdriBlanco0/worldcup2026-predictor)"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model accuracy (test 2018-2022)", "50.0%")
c2.metric("Baseline beaten by", "+12.5 pts")
c3.metric("Matches predicted", len(pred))
c4.metric("Matchdays tracked", "1 / 3")

tab_pred, tab_scores, tab_odds, tab_teams, tab_model = st.tabs(
    ["🔮 Predictions", "🎯 Exact Scores", "🏆 Tournament Odds", "🌍 Teams", "🤖 The Model"]
)


# ───────────────────────── TAB 1: PREDICTIONS ─────────────────────────
with tab_pred:
    st.subheader("Next matches")
    st.caption(
        "**Model: Random Forest v0** (trained on World Cup matches 1962-2022) · "
        "🟢 home win · 🟡 draw · 🔴 away win — kickoff times in Spanish time (CEST)"
    )

    dates = sorted(pred["date"].dt.date.unique())
    selected_date = st.selectbox("Pick a date", dates)
    day_matches = pred[pred["date"].dt.date == selected_date].sort_values("kickoff_spain")

    for _, m in day_matches.iterrows():
        with st.container(border=True):
            left, right = st.columns([2, 3])
            left.markdown(f"### {m['home_team']} 🆚 {m['away_team']}")
            left.caption(f"Group {m['group'][-1]} · {m['kickoff_spain'].strftime('%d %b · %H:%M')} 🇪🇸")
            right.markdown(f"**Model pick: {m['prediction']}**")
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
    st.caption(
        "**10,000 Monte Carlo simulations** of the full tournament — group stage, best thirds, "
        "and the real knockout bracket — powered by the Poisson goal model and current Elo ratings. "
        "Re-computed after every matchday."
    )

    odds = load_odds().sort_values("Champion", ascending=False)

    # Top contenders chart
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
    st.dataframe(
        odds,
        use_container_width=True, hide_index=True, height=600,
        column_config={
            "team": "Team",
            "R32": st.column_config.ProgressColumn("Round of 32", min_value=0, max_value=100, format="%.1f%%"),
            "R16": st.column_config.ProgressColumn("Round of 16", min_value=0, max_value=100, format="%.1f%%"),
            "QF": st.column_config.ProgressColumn("Quarter-final", min_value=0, max_value=100, format="%.1f%%"),
            "SF": st.column_config.ProgressColumn("Semi-final", min_value=0, max_value=100, format="%.1f%%"),
            "Final": st.column_config.ProgressColumn("Final", min_value=0, max_value=100, format="%.1f%%"),
            "Champion": st.column_config.ProgressColumn("🏆 Champion", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    st.info(
        "💡 Note: this model is more bullish on the favourites than betting markets "
        "(its Poisson slope is steep and knockout shootouts are Elo-weighted). The ranking "
        "matches the consensus; the magnitudes are the model's own opinion.",
        icon="📊",
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


# ───────────────────────── TAB 3: THE MODEL ─────────────────────────
with tab_model:
    st.subheader("How the model works")
    st.markdown("""
    | | |
    |---|---|
    | **Algorithm** | Random Forest (300 trees, max depth 6) |
    | **Training data** | All men's World Cup group-stage matches 1962-2022 (636 matches) |
    | **Features** | Historical win rate, goals for/against, World Cup experience, recent form (last 2 WCs), host advantage |
    | **Evaluation** | Temporal split: trained on 1962-2014, tested on 2018-2022 |
    | **Test accuracy** | **50.0%** vs 37.5% baseline (always picking "home win") |

    **No data leakage:** every match's features are computed using only matches played *before* that
    tournament — the model never sees the future.

    **Known limitations (v0):**
    - Never predicts draws (minority class: only ~25% of matches)
    - 2026 has 3 hosts for the first time — host advantage may be diluted
    - Defunct nations mapped to FIFA heirs (West Germany → Germany, Soviet Union → Russia...)

    **Coming in v1:** Elo ratings computed from scratch, squad features (age, caps, tournament
    experience) and draw-aware calibration.
    """)

    st.subheader("📈 Model performance during the tournament")
    st.info("The tournament starts on June 11. After every matchday, real results and model accuracy will be published here — hits and misses alike.")

st.divider()
st.caption("Built by Adrián Blanco · [LinkedIn](https://www.linkedin.com/in/adrianblancoajenjo/) · [GitHub](https://github.com/AdriBlanco0/worldcup2026-predictor) · Predictions are published before each matchday and never edited.")
