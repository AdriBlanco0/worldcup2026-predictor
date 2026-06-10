# ⚽ World Cup 2026 Predictor

> ¿Puede un modelo de Machine Learning predecir los resultados del Mundial mejor que un aficionado?

Proyecto de Data Science **en directo** durante el Mundial 2026 (Canadá · USA · México, 12 jun – 19 jul). Las predicciones se publican **antes** de cada jornada y el rendimiento del modelo se muestra públicamente: si falla, se verá. Si acierta, también.

## 🎯 El proyecto

Pipeline completo de Data Science, de principio a fin:

```
📊 DATOS  →  🗄️ SQL  →  🔍 EDA  →  🤖 MACHINE LEARNING  →  📈 STREAMLIT
```

| Fase | Qué se hace | Herramientas |
|------|-------------|--------------|
| **1. Datos** | Históricos de mundiales (1930-2022), convocatorias de los 48 equipos, resultados en tiempo real | `requests`, `BeautifulSoup`, APIs |
| **2. Base de datos** | Modelo relacional con partidos, equipos, jugadores y estadísticas | `MySQL`, `mysql-connector` |
| **3. EDA** | Análisis exploratorio: tendencias históricas, patrones, visualizaciones | `pandas`, `matplotlib`, `seaborn` |
| **4. Modelo ML** | Clasificación multiclase (1/X/2) comparando varios algoritmos | `scikit-learn`, `XGBoost` |
| **5. Dashboard** | Web interactiva con predicciones, stats y simulador del torneo | `Streamlit`, `Plotly`, `Folium` |

## 📡 Fuentes de datos

- **Kaggle** — históricos de todos los mundiales desde 1930
- **Wikipedia** — convocatorias oficiales de las 48 selecciones (1.248 jugadores)
- **APIs en tiempo real** — resultados y estadísticas durante el torneo
- **FBref / Transfermarkt** — estadísticas avanzadas y valores de mercado

## 🗓️ Estado del proyecto

- [x] Investigación y validación de fuentes de datos
- [ ] Recolección de datos históricos y convocatorias
- [ ] Base de datos MySQL
- [ ] EDA histórico (1930-2022)
- [ ] Entrenamiento y comparación de modelos
- [ ] Dashboard Streamlit
- [ ] Deploy en Streamlit Cloud
- [ ] Predicciones en directo durante el torneo

## 📈 Rendimiento del modelo

| Jornada | Partidos | Aciertos | % Acierto |
|---------|----------|----------|-----------|
| _El torneo empieza el 12 de junio..._ | – | – | – |

## 👤 Autor

**Adrián Blanco** — Data Scientist

Sígueme el proyecto en [LinkedIn](https://www.linkedin.com/) — predicciones publicadas antes de cada jornada.

## 📄 Licencia

MIT — úsalo, cópialo, aprende con él.
