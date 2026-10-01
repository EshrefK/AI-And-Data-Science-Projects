import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).parent

SPORTS = [
    "air hockey", "ampute football", "archery", "arm wrestling", "axe throwing",
    "balance beam", "barell racing", "baseball", "basketball", "baton twirling",
    "bike polo", "billiards", "bmx", "bobsled", "bowling", "boxing", "bull riding",
    "bungee jumping", "canoe slamon", "cheerleading", "chuckwagon racing", "cricket",
    "croquet", "curling", "disc golf", "fencing", "field hockey", "figure skating men",
    "figure skating pairs", "figure skating women", "fly fishing", "football",
    "formula 1 racing", "frisbee", "gaga", "giant slalom", "golf", "hammer throw",
    "hang gliding", "harness racing", "high jump", "hockey", "horse jumping",
    "horse racing", "horseshoe pitching", "hurdles", "hydroplane racing", "ice climbing",
    "ice yachting", "jai alai", "javelin", "jousting", "judo", "lacrosse", "log rolling",
    "luge", "motorcycle racing", "mushing", "nascar racing", "olympic wrestling",
    "parallel bar", "pole climbing", "pole dancing", "pole vault", "polo", "pommel horse",
    "rings", "rock climbing", "roller derby", "rollerblade racing", "rowing", "rugby",
    "sailboat racing", "shot put", "shuffleboard", "sidecar racing", "ski jumping",
    "sky surfing", "skydiving", "snow boarding", "snowmobile racing", "speed skating",
    "steer wrestling", "sumo wrestling", "surfing", "swimming", "table tennis", "tennis",
    "track bicycle", "trapeze", "tug of war", "ultimate", "uneven bars", "volleyball",
    "water cycling", "water polo", "weightlifting", "wheelchair basketball",
    "wheelchair racing", "wingsuit flying",
]
BULLY = [
    "age", "ethnicity", "gender", "not_cyberbullying",
    "other_cyberbullying", "religion",
]
MASKS = ["with mask", "without mask"]
STUDENT_FEATURES = [
    "Hours Studied", "Previous Scores", "Extracurricular Activities",
    "Sleep Hours", "Sample Question Papers Practiced",
]


@st.cache_resource
def keras_model(path):
    return tf.keras.models.load_model(ROOT / path, compile=False)


@st.cache_resource
def sk_model(path):
    return joblib.load(ROOT / path)


def image_batch(upload, size):
    image = Image.open(upload).convert("RGB").resize(size)
    return np.expand_dims(np.array(image, dtype="float32"), 0), image


def kaggle_csv(folder, name):
    return next((Path.home() / ".cache" / "kagglehub" / "datasets" / folder).rglob(name))


@st.cache_data(show_spinner=False)
def rec_catalog(kind):
    from sklearn.preprocessing import LabelEncoder

    if kind == "MovieLens":
        ratings = pd.read_csv(kaggle_csv("shubhammehta21/movie-lens-small-latest-dataset/versions/1", "ratings.csv"))
        movies = pd.read_csv(kaggle_csv("shubhammehta21/movie-lens-small-latest-dataset/versions/1", "movies.csv"))
        ratings = ratings.merge(movies[["movieId", "title"]], on="movieId")
        user_col, item_col, title_col, table, key = "userId", "movieId", "title", movies, "movieId"
        ceiling = "5"
    elif kind == "Anime":
        ratings = pd.read_csv(kaggle_csv("CooperUnion/anime-recommendations-database/versions/1", "rating.csv"))
        ratings = ratings.query("rating >= 0").sample(n=200000, random_state=42)
        table = pd.read_csv(kaggle_csv("CooperUnion/anime-recommendations-database/versions/1", "anime.csv"))
        ratings = ratings.merge(table[["anime_id", "name"]], on="anime_id")
        user_col, item_col, title_col, key = "user_id", "anime_id", "name", "anime_id"
        ceiling = "10"
    else:
        ratings = pd.read_csv(kaggle_csv("zygmunt/goodbooks-10k/versions/5", "ratings.csv"))
        ratings = ratings.sample(n=200000, random_state=42)
        table = pd.read_csv(kaggle_csv("zygmunt/goodbooks-10k/versions/5", "books.csv"))
        ratings = ratings.merge(table[["id", "title"]], left_on="book_id", right_on="id")
        user_col, item_col, title_col, key = "user_id", "book_id", "title", "id"
        ceiling = "5"
    user_encoder = LabelEncoder().fit(ratings[user_col])
    item_encoder = LabelEncoder().fit(ratings[item_col])
    ratings["u"] = user_encoder.transform(ratings[user_col])
    ratings["i"] = item_encoder.transform(ratings[item_col])
    names = table.drop_duplicates(key).set_index(key)[title_col]
    titles = [str(names.get(item, item)) for item in item_encoder.classes_]
    counts = ratings.groupby("u").size().sort_values(ascending=False).head(30)
    return {
        "titles": titles,
        "ceiling": ceiling,
        "popular": int(ratings.groupby("i").size().idxmax()),
        "user_codes": [int(code) for code in counts.index],
        "user_labels": [
            f"Kullanıcı {int(user_encoder.classes_[code])} · {int(counts[code])} puan"
            for code in counts.index
        ],
    }


@st.cache_data(show_spinner=False)
def series_bundle(kind):
    from sklearn.preprocessing import MinMaxScaler

    if kind == "Delhi sıcaklık":
        folder = "sumanthvrao/daily-climate-time-series-data/versions/3"
        frame = pd.concat(
            [pd.read_csv(kaggle_csv(folder, "DailyDelhiClimateTrain.csv")), pd.read_csv(kaggle_csv(folder, "DailyDelhiClimateTest.csv"))],
            ignore_index=True,
        )
        frame = frame.drop_duplicates("date")
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date").set_index("date")
        raw = frame[["meantemp", "humidity", "wind_speed"]].to_numpy()
        split = int(len(raw) * 0.8)
        scaled = MinMaxScaler().fit(raw[:split]).transform(raw)
        target = MinMaxScaler().fit(raw[:split, :1])
        history = frame["meantemp"].tail(60)
        return {
            "path": "TimeSeries/delhi_temp_lstm.keras",
            "window": scaled[-14:].reshape(1, 14, 3).tolist(),
            "ymin": float(target.data_min_[0]),
            "ymax": float(target.data_max_[0]),
            "dates": [str(d) for d in history.index],
            "values": [float(v) for v in history.to_numpy()],
            "unit": "C",
            "label": "Sonraki günün ortalama sıcaklığı",
            "caption": "Son 14 günün sıcaklık, nem ve rüzgar değerinden bir sonraki gün tahmin edilir.",
        }
    if kind == "Elektrik yükü":
        frame = pd.read_csv(kaggle_csv("robikscube/hourly-energy-consumption/versions/3", "AEP_hourly.csv"))
        frame["Datetime"] = pd.to_datetime(frame["Datetime"])
        frame = frame.sort_values("Datetime").drop_duplicates("Datetime").set_index("Datetime").tail(24 * 365 * 2)
        raw = frame[["AEP_MW"]].to_numpy()
        split = int(len(raw) * 0.8)
        scaler = MinMaxScaler().fit(raw[:split])
        scaled = scaler.transform(raw).ravel()
        history = frame["AEP_MW"].tail(48)
        return {
            "path": "TimeSeries/aep_load_gru.keras",
            "window": scaled[-24:].reshape(1, 24, 1).tolist(),
            "ymin": float(scaler.data_min_[0]),
            "ymax": float(scaler.data_max_[0]),
            "dates": [str(d) for d in history.index],
            "values": [float(v) for v in history.to_numpy()],
            "unit": "MW",
            "label": "Sonraki saatin yükü",
            "caption": "Son 24 saatin elektrik yükünden bir sonraki saat tahmin edilir.",
        }
    frame = pd.read_csv(kaggle_csv("sudalairajkumar/cryptocurrencypricehistory/versions/3", "coin_Bitcoin.csv"))
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    raw = frame[["Close"]].to_numpy()
    scaler = MinMaxScaler().fit(raw)
    scaled = scaler.transform(raw).ravel()
    history = frame["Close"].tail(60)
    return {
        "path": "TimeSeries/bitcoin_cnn.keras",
        "window": scaled[-30:].reshape(1, 30, 1).tolist(),
        "ymin": float(scaler.data_min_[0]),
        "ymax": float(scaler.data_max_[0]),
        "dates": [str(d) for d in history.index],
        "values": [float(v) for v in history.to_numpy()],
        "unit": "USD",
        "label": "Sonraki günün kapanışı",
        "caption": "Son 30 kapanış fiyatından bir sonraki günün kapanışı tahmin edilir.",
    }


def show_text_result(title, body):
    st.subheader(title)
    st.write(body)


st.set_page_config(page_title="Saved models", layout="centered")
st.title("Kayıtlı modeller")
st.caption("Notebook'larda kaydedilen modeli yükler. Eğitimi yeniden çalıştırmaz.")

topic = st.sidebar.selectbox(
    "Konu",
    ["Regression", "Classification", "Computer Vision", "NLP", "Recommendation", "Time Series"],
)

if topic == "Regression":
    choice = st.selectbox("Model", ["Öğrenci performansı", "Araba fiyatı"])
    if choice == "Öğrenci performansı":
        st.caption("Kayıtlı dosya lineer model. Scaler yanında yoksa sonuç 0-100 indeks olmayabilir.")
        hours = st.number_input("Çalışılan saat", 0, 24, 7)
        previous = st.number_input("Önceki puan", 0, 100, 80)
        extra = st.selectbox("Ders dışı etkinlik", ["Hayır", "Evet"])
        sleep = st.number_input("Uyku saati", 0, 24, 7)
        papers = st.number_input("Çözülen deneme", 0, 20, 3)
        if st.button("Tahmin et", type="primary"):
            model = sk_model("Regression/Regression3student_performance_model.joblib")
            row = [[hours, previous, 1 if extra == "Evet" else 0, sleep, papers]]
            show_text_result("Performance Index", f"{float(model.predict(row)[0]):.1f}")
    else:
        pack = sk_model("Regression/Regression2car_price_model.joblib")
        model, columns = pack["model"], pack["columns"]
        enginesize = st.number_input("Motor hacmi", 50, 400, 130)
        citympg = st.number_input("Şehir içi mil", 5, 60, 25)
        car_volume = st.number_input("Hacim", 1.0, 40.0, 10.0)
        brands = sorted(c.replace("brand_", "") for c in columns if c.startswith("brand_"))
        brand = st.selectbox("Marka", brands)
        if st.button("Tahmin et", type="primary"):
            row = {c: 0.0 for c in columns}
            row["enginesize"] = enginesize
            row["citympg"] = citympg
            row["car_volume"] = car_volume
            row["fueltype_gas"] = 1
            key = "brand_" + brand
            if key in row:
                row[key] = 1
            price = float(model.predict(np.array([[row[c] for c in columns]]))[0])
            show_text_result("Tahmini fiyat", f"{price:,.0f}")

elif topic == "Classification":
    choice = st.selectbox("Model", ["Cam türü", "Sigorta ilgisi", "Kredi kartı"])
    if choice == "Cam türü":
        names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
        defaults = [1.52, 13.0, 2.0, 1.2, 72.0, 0.5, 9.0, 0.0, 0.0]
        values = [st.number_input(n, value=d, format="%.4f") for n, d in zip(names, defaults)]
        if st.button("Tahmin et", type="primary"):
            model = sk_model("Classification/Classification3Glass_prediction.joblib")
            pred = model.predict(np.array([values]))[0]
            show_text_result("Cam sınıfı", str(pred))
    elif choice == "Sigorta ilgisi":
        gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        age = st.number_input("Yaş", 18, 90, 30)
        license_held = st.selectbox("Ehliyet", ["Var", "Yok"])
        region = st.number_input("Bölge kodu", 0, 52, 28)
        insured = st.selectbox("Daha önce sigortalı", ["Hayır", "Evet"])
        vehicle_age = st.selectbox("Araç yaşı", ["1 yıldan az", "1-2 yıl", "2 yıldan fazla"])
        damage = st.selectbox("Hasar geçmişi", ["Yok", "Var"])
        premium = st.number_input("Yıllık prim", 0, 100000, 30000)
        channel = st.number_input("Satış kanalı", 0, 200, 152)
        vintage = st.number_input("Müşteri süresi (gün)", 0, 400, 150)
        if st.button("Tahmin et", type="primary"):
            model = sk_model("Classification/insurance_model.joblib")
            age_code = {"1 yıldan az": 0, "1-2 yıl": 1, "2 yıldan fazla": 2}[vehicle_age]
            row = [[
                1 if gender == "Erkek" else 0,
                age,
                1 if license_held == "Var" else 0,
                region,
                1 if insured == "Evet" else 0,
                age_code,
                1 if damage == "Var" else 0,
                premium,
                channel,
                vintage,
            ]]
            prob = float(model.predict_proba(row)[0, 1])
            show_text_result("İlgi olasılığı", f"{prob:.3f}")
    else:
        st.caption("V1–V28 ve Amount. Modelin içindeki scaler ham değerleri ölçekler.")
        amount = st.number_input("Amount", 0.0, 10000.0, 50.0)
        vs = [st.number_input(f"V{i}", value=0.0, format="%.3f", key=f"v{i}") for i in range(1, 29)]
        if st.button("Tahmin et", type="primary"):
            model = sk_model("Classification/Classification1creditcard.joblib")
            row = vs + [amount]
            pred = int(model.predict([row])[0])
            prob = float(model.predict_proba([row])[0, 1])
            show_text_result("Sonuç", "fraud" if pred == 1 else "normal")
            st.write(f"Fraud olasılığı: {prob:.4f}")

elif topic == "Computer Vision":
    choice = st.selectbox("Model", ["Yüz maskesi", "Spor"])
    source = st.radio("Girdi", ["Fotoğraf", "Kamera"], horizontal=True)
    shot = st.file_uploader("Fotoğraf", type=["jpg", "jpeg", "png"]) if source == "Fotoğraf" else st.camera_input("Kamera")
    if st.button("Tahmin et", type="primary") and shot is not None:
        if choice == "Yüz maskesi":
            batch, preview = image_batch(shot, (128, 128))
            model = keras_model("ComputerVision/facemask.keras")
            pred = model(batch, training=False).numpy()[0]
            st.image(preview, width=240)
            show_text_result("Sınıf", MASKS[int(pred.argmax())])
            st.write({MASKS[i]: f"{float(pred[i]):.3f}" for i in range(len(MASKS))})
        else:
            batch, preview = image_batch(shot, (224, 224))
            model = keras_model("ComputerVision/sports_cnn.keras")
            pred = model(batch, training=False).numpy()[0]
            order = np.argsort(-pred)[:3]
            st.image(preview, width=240)
            show_text_result("Spor", SPORTS[int(order[0])])
            st.write({SPORTS[int(i)]: f"{float(pred[i]):.3f}" for i in order})

elif topic == "NLP":
    choice = st.selectbox("Model", ["Siber zorbalık", "Sahte haber", "Film yorumu"])
    text = st.text_area("Metin", height=140)
    if st.button("Tahmin et", type="primary") and text.strip():
        if choice == "Siber zorbalık":
            model = keras_model("NLP/cyberbullying.keras")
            pred = model(tf.constant([text]), training=False).numpy()[0]
            show_text_result("Sınıf", BULLY[int(pred.argmax())])
        elif choice == "Sahte haber":
            model = keras_model("NLP/fake_news.keras")
            prob = float(model(tf.constant([text]), training=False).numpy()[0, 0])
            show_text_result("Sınıf", "true" if prob >= 0.5 else "fake")
            st.write(f"True olasılığı: {prob:.3f}")
        else:
            model = keras_model("NLP/movie_sentiment.keras")
            prob = float(model(tf.constant([text]), training=False).numpy()[0, 0])
            show_text_result("Duygu", "positive" if prob >= 0.5 else "negative")
            st.write(f"Positive olasılığı: {prob:.3f}")

elif topic == "Recommendation":
    choice = st.selectbox("Model", ["MovieLens", "Anime", "Kitap"])
    paths = {
        "MovieLens": "RecommendationSystems/movielens_ncf.keras",
        "Anime": "RecommendationSystems/anime_ncf.keras",
        "Kitap": "RecommendationSystems/books_ncf.keras",
    }
    with st.spinner("Başlıklar yükleniyor"):
        catalog = rec_catalog(choice)
    st.caption(f"Puan aralığı 1–{catalog['ceiling']}. Listede en çok puan vermiş kullanıcılar var.")
    user_pick = st.selectbox(
        "Kullanıcı",
        range(len(catalog["user_labels"])),
        format_func=lambda i: catalog["user_labels"][i],
    )
    item_pick = st.selectbox(
        "Başlık",
        range(len(catalog["titles"])),
        index=catalog["popular"],
        format_func=lambda i: catalog["titles"][i],
    )
    if st.button("Tahmin et", type="primary"):
        model = keras_model(paths[choice])
        user = catalog["user_codes"][user_pick]
        one = model([np.array([[user]]), np.array([[item_pick]])], training=False).numpy()[0, 0]
        show_text_result(catalog["titles"][item_pick], f"{float(one):.2f} / {catalog['ceiling']}")
        count = len(catalog["titles"])
        scores = model(
            [np.full((count, 1), user), np.arange(count).reshape(-1, 1)],
            training=False,
        ).numpy().ravel()
        st.write("Bu kullanıcı için en yüksek 5 başlık")
        for rank, index in enumerate(np.argsort(-scores)[:5], start=1):
            st.write(f"{rank}. {catalog['titles'][int(index)]} — {float(scores[index]):.2f}")

else:
    choice = st.selectbox("Model", ["Delhi sıcaklık", "Elektrik yükü", "Bitcoin"])
    with st.spinner("Seri yükleniyor"):
        bundle = series_bundle(choice)
    st.caption(bundle["caption"])
    st.line_chart(pd.DataFrame({"değer": bundle["values"]}, index=pd.to_datetime(bundle["dates"])))
    if st.button("Sonraki değeri tahmin et", type="primary"):
        model = keras_model(bundle["path"])
        scaled = float(model(np.array(bundle["window"], dtype="float32"), training=False).numpy()[0, 0])
        value = scaled * (bundle["ymax"] - bundle["ymin"]) + bundle["ymin"]
        show_text_result(bundle["label"], f"{value:,.2f} {bundle['unit']}")

st.sidebar.caption("Ajanlar ayrı uygulamada: AIAgents/app.py. Kümeleme, SMS ve Intel sahne modeli bu klasörde kayıtlı değil. Taksi modeli scaler ile birlikte kaydedilmedi.")
