import streamlit as st
import pandas as pd
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

pred = load_predictions()
squads = load_squads()


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

tab_pred, tab_teams, tab_model = st.tabs(["🔮 Predictions", "🌍 Teams", "🤖 The Model"])


# ───────────────────────── TAB 1: PREDICTIONS ─────────────────────────
with tab_pred:
    st.subheader("Next matches")
    st.caption("🟢 home win · 🟡 draw · 🔴 away win — kickoff times in Spanish time (CEST)")

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


# ───────────────────────── TAB 2: TEAMS ─────────────────────────
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
