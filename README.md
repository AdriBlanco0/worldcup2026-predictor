# ⚽ World Cup 2026 Predictor

> Can a Machine Learning model predict World Cup results better than a football fan?

### 🔴 **[LIVE DASHBOARD → predictor-worldcup2026.streamlit.app](https://predictor-worldcup2026.streamlit.app)**

A **live** Data Science project running throughout the 2026 World Cup (Canada · USA · Mexico, Jun 11 – Jul 19). Predictions are published **before** every matchday and the model's performance is tracked publicly: if it fails, everyone will see it. If it succeeds, too.

## 🎯 The project

End-to-end Data Science pipeline:

```
📊 DATA  →  🗄️ SQL  →  🔍 EDA  →  🤖 MACHINE LEARNING  →  📈 STREAMLIT
```

| Phase | What happens | Tools |
|-------|--------------|-------|
| **1. Data** | World Cup history (1930-2022), squads of all 48 teams, real-time results | `requests`, `BeautifulSoup`, APIs |
| **2. Database** | Relational model with matches, teams, players and stats | `MySQL`, `mysql-connector` |
| **3. EDA** | Exploratory analysis: historical trends, patterns, visualizations | `pandas`, `matplotlib`, `seaborn` |
| **4. ML Model** | Multiclass classification (home win / draw / away win) comparing several algorithms | `scikit-learn`, `XGBoost` |
| **5. Dashboard** | Interactive web app with predictions, stats and tournament simulator | `Streamlit`, `Plotly`, `Folium` |

## 📡 Data sources

- **Kaggle** — historical data from every World Cup since 1930
- **Wikipedia** — official squads of all 48 national teams (1,248 players)
- **Real-time APIs** — results and statistics during the tournament
- **FBref / Transfermarkt** — advanced stats and market values

## 🗓️ Project status

- [x] Research and validation of data sources
- [x] Historical data and squads collection
- [x] Historical EDA (1930-2022) — see [notebook 01](notebooks/01_eda_historical.ipynb)
- [ ] Feature engineering and model v0
- [ ] MySQL database
- [ ] Model v1 (Elo ratings + squad features) and comparison
- [x] Streamlit dashboard
- [x] Deploy on Streamlit Cloud — [live app](https://predictor-worldcup2026.streamlit.app)
- [ ] Live predictions during the tournament

## 🧭 Roadmap — Player Performance Tracker

Beyond match predictions, the project will include a **player performance module** during the tournament:

- **Tournament form index:** which players are over- or under-performing their expected level
  (based on their historical caps/goals ratio and market value) — the tournament's *revelations*
  and *disappointments*, updated after every matchday.
- **Tournament vs club season** *(after the group stage)*: comparison of each player's World Cup
  performance against their 2025/26 club season stats.
- **Group stage vs knockout stage** *(after the group stage)*: which players raise their level
  when elimination is on the line — and which ones disappear in big matches.

## 📈 Model performance

| Matchday | Matches | Correct | Accuracy |
|----------|---------|---------|----------|
| _Tournament starts June 11..._ | – | – | – |

## 👤 Author

**Adrián Blanco** — Data Scientist

Follow the project on [LinkedIn](https://www.linkedin.com/in/adrianblancoajenjo/) — predictions published before every matchday.

## 📄 License

MIT — use it, copy it, learn from it.
