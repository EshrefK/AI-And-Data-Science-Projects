import glob
import json
import os
from pathlib import Path

import chromadb
import kagglehub
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).parent
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

AGENTS = {
    "Kitap danışmanı": {
        "example": "I want a sad literary novel, not a long fantasy series",
        "prompt": (
            "You are a book advisor. Use ONLY the books in the context.\n"
            "Format: ## Picks then a bullet for each book with the title and why it fits.\n\n"
            "Context:\n{context}\n\nReader:\n{query}"
        ),
    },
    "Ev danışmanı": {
        "example": "A 2 bedroom unit under 700000 close to the city center",
        "prompt": (
            "You are a house buyer advisor. Use ONLY the sales in the context. Prices are AUD.\n"
            "Format: ## Best matches then a bullet with suburb, type, rooms, price and distance.\n"
            "## Final pick with one sentence.\n\n"
            "Context:\n{context}\n\nBuyer:\n{query}"
        ),
    },
    "Film danışmanı": {
        "example": "A children's animated comedy",
        "prompt": (
            "You are a movie advisor. Use ONLY the movies in the context.\n"
            "Format: ## Picks then a bullet for each movie with the title and why it fits.\n\n"
            "Context:\n{context}\n\nViewer:\n{query}"
        ),
    },
}


def api_key():
    text = (ROOT / ".env").read_text(encoding="utf-8")
    return text.split("=", 1)[1].strip().strip('"').strip("'")


def book_rows():
    path = kagglehub.dataset_download("zygmunt/goodbooks-10k")
    books = pd.read_csv(glob.glob(os.path.join(path, "**", "books.csv"), recursive=True)[0])
    tags = pd.read_csv(glob.glob(os.path.join(path, "**", "tags.csv"), recursive=True)[0])
    book_tags = pd.read_csv(glob.glob(os.path.join(path, "**", "book_tags.csv"), recursive=True)[0])
    book_tags = book_tags.merge(tags, on="tag_id")
    book_tags = book_tags[book_tags["tag_name"].str.fullmatch(r"[a-z][a-z\-]{2,24}")]
    shelf = ["to-read", "favorites", "currently-reading", "owned", "books-i-own", "favourites", "kindle", "ebook", "ebooks", "my-books", "to-buy", "owned-books"]
    book_tags = book_tags[~book_tags["tag_name"].isin(shelf)]
    top_tags = book_tags.sort_values("count", ascending=False).groupby("goodreads_book_id").head(6)
    tag_text = top_tags.groupby("goodreads_book_id")["tag_name"].agg(" ".join)
    books = books.merge(tag_text.rename("tags"), left_on="book_id", right_index=True)
    books = books.sample(2500, random_state=42)
    year = books["original_publication_year"].fillna(0).astype(int).astype(str)
    text = (
        books["title"].fillna("")
        + " | " + books["authors"].fillna("")
        + " | " + year
        + " | rating " + books["average_rating"].astype(str)
        + " | " + books["tags"].fillna("")
    )
    return books["book_id"].astype(str).tolist(), text.tolist()


def house_rows():
    path = kagglehub.dataset_download("dansbecker/melbourne-housing-snapshot")
    df = pd.read_csv(glob.glob(os.path.join(path, "**", "melb_data.csv"), recursive=True)[0])
    df["type_name"] = df["Type"].map({"h": "house", "u": "unit", "t": "townhouse"})
    df = df.dropna(subset=["Price", "Rooms", "Bathroom", "Distance", "Regionname", "type_name"])
    df = df.sample(2500, random_state=42).reset_index(drop=True)
    text = (
        df["Suburb"]
        + " | " + df["type_name"]
        + " | rooms " + df["Rooms"].astype(int).astype(str)
        + " | bathrooms " + df["Bathroom"].astype(int).astype(str)
        + " | price " + df["Price"].astype(int).astype(str)
        + " | distance_km " + df["Distance"].astype(str)
        + " | " + df["Regionname"]
    )
    return df.index.astype(str).tolist(), text.tolist()


def movie_rows():
    path = kagglehub.dataset_download("shubhammehta21/movie-lens-small-latest-dataset")
    movies = pd.read_csv(glob.glob(os.path.join(path, "**", "movies.csv"), recursive=True)[0])
    movies = movies[movies["genres"] != "(no genres listed)"]
    movies = movies.sample(2500, random_state=42)
    text = movies["title"] + " | " + movies["genres"].str.replace("|", ", ", regex=False)
    return movies["movieId"].astype(str).tolist(), text.tolist()


BUILDERS = {
    "Kitap danışmanı": ("books", book_rows),
    "Ev danışmanı": ("houses", house_rows),
    "Film danışmanı": ("movies", movie_rows),
}


@st.cache_resource(show_spinner="Kayıtlar hazırlanıyor. İlk seferde bir iki dakika sürebilir.")
def collection(agent_name):
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection_name, build_rows = BUILDERS[agent_name]
    col = client.get_or_create_collection(collection_name)
    if col.count() == 0:
        ids, docs = build_rows()
        col.add(ids=ids, documents=docs)
    return col


def ask(agent_name, query):
    col = collection(agent_name)
    found = col.query(query_texts=[query], n_results=5)
    docs = found["documents"][0]
    prompt = AGENTS[agent_name]["prompt"].format(context="\n".join(docs), query=query)
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key(),
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }),
        timeout=90,
    )
    body = response.json()
    if "choices" not in body:
        return docs, body.get("error", {}).get("message", response.text)
    return docs, body["choices"][0]["message"]["content"]


st.set_page_config(page_title="AI Agents", layout="centered")
st.title("AI Agents")
st.caption("Ajanı seç, soruyu yaz. Cevap sadece Chroma'nın bulduğu kayıtlardan gelir.")

agent_name = st.selectbox("Ajan", list(AGENTS))
query = st.text_area("Soru", value=AGENTS[agent_name]["example"], height=90, key=agent_name)

if st.button("Sor", type="primary"):
    docs, answer = ask(agent_name, query)
    st.subheader("Bulunan kayıtlar")
    st.text("\n\n".join(docs))
    st.subheader("Cevap")
    st.markdown(answer)
