import pandas as pd
import numpy as np
import pickle
import re 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

print("Loading Dataset...")

movies=pd.read_csv("dataset/TMDB_all_movies.csv")
print(f"Original Movies: {len(movies)}")

movies = movies[~movies["genres"].str.contains("Documentary",case=False,na=False)]

movies=movies[
    [
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "director",
        "writers",
        "vote_average",
        "vote_count",
        "imdb_rating",
        "imdb_votes",
        "popularity",
        "poster_path",
        "runtime",
        "release_date",
    ]
]

movies.drop_duplicates(subset="title",inplace=True)

text_columns=[
    "overview",
    "genres",
    "keywords",
    "cast",
    "director",
    "writers",
]

for col in text_columns:
    movies[col]=movies[col].fillna("")

def top_cast(cast):
    cast = str(cast).split(",")
    return " ".join(cast[:5])

movies["cast"] = movies["cast"].apply(top_cast)

numeric_columns = [
    "vote_average",
    "vote_count",
    "imdb_rating",
    "imdb_votes",
    "popularity",
]

for col in numeric_columns:
    movies[col]=pd.to_numeric(movies[col],errors="coerce").fillna(0)

movies=movies[movies["vote_count"]>=50]
print(f"Movies after filtering: {len(movies)}")

def clean_text(text):
    text=str(text).lower()
    text=text.replace("|"," ")
    text=text.replace(","," ")
    text=re.sub(r"[^a-z0-9 ]"," ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

movies["overview"] = movies["overview"].apply(
    lambda x: " ".join(str(x).split()[:50])
)

movies["tags"]=(
    movies["genres"] + " "
    + movies["genres"]+ " "
    + movies["genres"]+ " "
    + movies["keywords"] + " "
    + movies["keywords"] + " "
    + movies["cast"] + " "
    + movies["director"] + " "
    + movies["director"] + " "
    + movies["writers"] + " "
    + movies["overview"]
)

movies["tags"] = movies["tags"].apply(clean_text)
movies = movies[movies["tags"].str.len() > 20]

print("Creating TF-IDF vectors...")

tfidf=TfidfVectorizer(stop_words="english",max_features=20000,ngram_range=(1,2),min_df=2)

vectors=tfidf.fit_transform(movies["tags"])
print("Vector shape :",vectors.shape)

nn_model=NearestNeighbors(metric="cosine",algorithm="brute",n_neighbors=20)

nn_model.fit(vectors)

scaler = MinMaxScaler()

movies["vote_avg_scaled"] = scaler.fit_transform(
    movies[["vote_average"]]
)
movies["vote_count_scaled"] = scaler.fit_transform(
    movies[["vote_count"]]
)
movies["imdb_scaled"] = scaler.fit_transform(
    movies[["imdb_rating"]]
)
movies["popularity_scaled"] = scaler.fit_transform(
    movies[["popularity"]]
)

movies["hybrid_popularity"] = (
    0.35 * movies["vote_avg_scaled"]
    + 0.25 * movies["imdb_scaled"]
    + 0.20 * movies["vote_count_scaled"]
    + 0.20 * movies["popularity_scaled"]
)

movies.reset_index(drop=True, inplace=True)
pickle.dump(movies,open("movies.pkl", "wb"))
pickle.dump(tfidf,open("tfidf.pkl", "wb"))
pickle.dump(vectors,open("vectors.pkl", "wb"))
pickle.dump(nn_model,open("nn_model.pkl", "wb"))
print("\nTraining Completed Successfully!")
print(f"Movies Saved : {len(movies)}")
print("Generated Files:")
print("   movies.pkl")
print("   tfidf.pkl")
print("   vectors.pkl")
print("   nn_model.pkl")