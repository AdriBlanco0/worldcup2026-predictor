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
| **2. Database** | Normalized MySQL schema (teams, matches, predictions, odds, players) loaded from the CSVs; analytical SQL (JOINs, CTEs, subqueries, views) | `MySQL`, `mysql-connector` |
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
- [x] Feature engineering and model v0 (Random Forest)
- [x] Model v1 (self-computed Elo ratings) + leave-one-tournament-out validation
- [x] Poisson exact-score model + Monte Carlo tournament simulator
- [x] Probabilistic evaluation (RPS/Brier) + RF·Poisson ensemble
- [x] Streamlit dashboard — [live app](https://predictor-worldcup2026.streamlit.app)
- [x] Live predictions + daily auto-update pipeline during the tournament
- [x] MySQL relational database (5 tables, normalized) + analytical SQL — see [notebook 07](notebooks/07_sql_analysis.ipynb)
- [ ] Player Performance Tracker

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

Updated after every matchday — hits and misses alike.

<!-- PERFORMANCE_START -->
| Matches | v0 (outcome) | v1 (outcome) | 🎯 Exact scores | 📐 RPS (v1) |
|---|---|---|---|---|
| 24 | 11/24 (46%) | 9/24 (38%) | 4/24 (17%) | 0.211 |

🎯 **Exact scores nailed:** Mexico 2-0 South Africa, Brazil 1-1 Morocco, Haiti 0-1 Scotland, Belgium 1-1 Egypt.
<!-- PERFORMANCE_END -->

The Poisson model is hitting exact scores well above the ~11% that world-class exact-score
models achieve. (Small sample; expect regression toward the mean as the tournament goes on.)

📐 **On metrics:** accuracy is the wrong yardstick for a probabilistic forecaster. By the
**Ranked Probability Score** (the standard metric for ordered football outcomes, ~0.20 for this
model historically), the model performs close to expectations even when raw accuracy looks noisy
over a small sample — because it assigns honest probability to draws even when they aren't the top
pick. See [notebook 06](notebooks/06_probabilistic_eval_ensemble.ipynb) for the full analysis and the
RF·Poisson ensemble that wins on RPS.

## 👤 Author

**Adrián Blanco** — Data Scientist

Follow the project on [LinkedIn](https://www.linkedin.com/in/adrianblancoajenjo/) — predictions published before every matchday.

## 📄 License

MIT — use it, copy it, learn from it.
