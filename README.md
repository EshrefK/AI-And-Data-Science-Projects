# AI and data science projects

Practice work grouped by task. Each folder is a notebook plus, where it fit on GitHub, the saved model. `streamlit_app.py` is a single demo over these projects.

This is course and self-study work. Any score in a notebook is my own hold-out check, not a public leaderboard result.

## Folders

| Folder | What it is |
|---|---|
| Regression | Taxi fare, car price, student performance |
| Classification | Credit card, glass type, health insurance |
| Clustering | Spotify tracks, life expectancy, city lifestyle |
| ComputerVision | Sports images, Intel scenes, face-mask detection |
| NLP | SMS spam, fake news, movie-review sentiment, cyberbullying |
| RecommendationSystems | MovieLens, anime, books |
| TimeSeries | Delhi temperature, electricity load, Bitcoin price |
| AIAgents | RAG advisors for books, houses, and movies. Chroma embeds each row and retrieves the closest ones. A chat model writes the answer from those rows only. `app.py` runs the demo. |

## Run the demo

```bash
pip install streamlit pandas scikit-learn
streamlit run streamlit_app.py
```

Large raw datasets stay out of the repository. A wider set of past Kaggle competitions is in [KaggleCompitions](https://github.com/EshrefK/KaggleCompitions).
