# ⚽ World Cup 2026 Predictor

> Can a Machine Learning model predict World Cup results better than a football fan?

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
- [ ] Historical data and squads collection
- [ ] MySQL database
- [ ] Historical EDA (1930-2022)
- [ ] Model training and comparison
- [ ] Streamlit dashboard
- [ ] Deploy on Streamlit Cloud
- [ ] Live predictions during the tournament

## 📈 Model performance

| Matchday | Matches | Correct | Accuracy |
|----------|---------|---------|----------|
| _Tournament starts June 11..._ | – | – | – |

## 👤 Author

**Adrián Blanco** — Data Scientist

Follow the project on [LinkedIn](https://www.linkedin.com/in/adrianblancoajenjo/) — predictions published before every matchday.

## 📄 License

MIT — use it, copy it, learn from it.
