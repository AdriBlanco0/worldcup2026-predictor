import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

# Robust path: works locally and on Streamlit Cloud
DATA = Path(__file__).parent.parent / "data" / "processed"

@st.cache_data
def load_predictions():
    df = pd.read_csv(DATA / "predictions_2026_group_stage_v0.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

pred = load_predictions()

# ───────────────────────── HEADER ─────────────────────────
st.title("⚽ World Cup 2026 Predictor")
st.markdown(
    "**Machine Learning predictions for every match — published BEFORE each matchday.** "
    "Model v0: Random Forest trained on World Cups 1962-2022. "
    "[Code on GitHub](https://github.com/AdriBlanco0/worldcup2026-predictor)"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model accuracy (test 2018-2022)", "50.0%")
col2.metric("Baseline beaten by", "+12.5 pts")
col3.metric("Matches predicted", len(pred))
col4.metric("Matchdays tracked", "1 / 3")

st.divider()

# ───────────────────────── TODAY'S MATCHES ─────────────────────────
st.header("🔮 Next matches")

dates = sorted(pred["date"].dt.date.unique())
selected_date = st.selectbox("Pick a date", dates)

day_matches = pred[pred["date"].dt.date == selected_date]

for _, m in day_matches.iterrows():
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        c1.subheader(f"{m['home_team']} 🆚 {m['away_team']}")
        c1.caption(f"Group {m['group'][-1]} · {m['date'].date()}")
        c2.metric(f"{m['home_team']} win", f"{m['p_home_win']}%")
        c3.metric("Draw", f"{m['p_draw']}%")
        c4.metric(f"{m['away_team']} win", f"{m['p_away_win']}%")


        st.divider()

# ───────────────────────── ALL PREDICTIONS ─────────────────────────
st.header("📊 All group-stage predictions")

groups = ["All"] + sorted(pred["group"].unique())
selected_group = st.selectbox("Filter by group", groups)

table = pred if selected_group == "All" else pred[pred["group"] == selected_group]
st.dataframe(
    table[["date", "group", "home_team", "away_team",
           "p_home_win", "p_draw", "p_away_win", "prediction"]],
    use_container_width=True, hide_index=True,
)

st.divider()

# ───────────────────────── ABOUT ─────────────────────────
with st.expander("🤖 How does the model work?"):
    st.markdown("""
    - **Algorithm:** Random Forest (300 trees) trained on all men's World Cups 1962-2022
    - **Features:** historical win rate, goals for/against, World Cup experience,
      recent form (last 2 World Cups) and host advantage
    - **No data leakage:** every match's features use only data available *before* that match
    - **Honest evaluation:** temporal split — trained on 1962-2014, tested on 2018-2022 (50% vs 37.5% baseline)
    - **Known limitation:** v0 never predicts draws (minority class). v1 with Elo ratings coming soon.
    """)

st.caption("Built by Adrián Blanco · [LinkedIn](https://www.linkedin.com/in/adrianblancoajenjo/) · Predictions are published before each matchday and never edited.")