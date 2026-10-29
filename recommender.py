import pickle
import pandas as pd
import numpy as np
from difflib import get_close_matches

movies = pickle.load(open("movies.pkl", "rb"))
vectors = pickle.load(open("vectors.pkl", "rb"))
model = pickle.load(open("nn_model.pkl", "rb"))

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
def get_poster(poster_path):
    if pd.isna(poster_path) or str(poster_path).strip()=="":
        return None
    return POSTER_BASE_URL+poster_path

def find_movie(title):
    title=title.lower().strip()
    titles=movies["title"].str.lower().tolist()
    if title in titles:
        return titles.index(title)

    match=get_close_matches(title,titles,n=1,cutoff=0.6)
    if len(match)==0:
        return None
    return titles.index(match[0])

def recommend(movie_name,n_recommendations=10):
    movie_index=find_movie(movie_name)
    if movie_index is None:
        return []
    n_neighbors = min(100, len(movies))
    distances,indices=model.kneighbors(vectors[movie_index],n_neighbors=n_neighbors)
    recommendations=[]
    for distance,idx in zip(distances[0],indices[0]):
        if idx==movie_index:
            continue
        similarity=1-distance
        row=movies.iloc[idx]
        hybrid_score=(
            0.80 * similarity
            + 0.08 * row["imdb_scaled"]
            + 0.07 * row["vote_avg_scaled"]
            + 0.05 * row["popularity_scaled"]
        )

        recommendations.append(
            {
                "title": row["title"],
                "poster": get_poster(row["poster_path"]),
                "genres": row["genres"],
                "runtime": row["runtime"],
                "release_date": row["release_date"],
                "tmdb_rating": row["vote_average"],
                "imdb_rating": row["imdb_rating"],
                "similarity": round(similarity * 100, 2),
                "hybrid_score": round(hybrid_score * 100, 2),
                "overview": row["overview"],
                "director": row["director"],
                "cast": row["cast"],
            }
        )
    recommendations = sorted(
    recommendations,
    key=lambda x: x["hybrid_score"],
    reverse=True,
    )

    return recommendations[:n_recommendations]

def search_movies(query,limit=10):
    query=query.lower()
    results=movies[movies["title"].str.lower().str.contains(query,na=False)].sort_values("hybrid_popularity",ascending=False)["title"].tolist()
    return results[:limit]

def get_movie_details(movie_name):
    idx = find_movie(movie_name)
    if idx is None:
        return None
    row = movies.iloc[idx]   
    return {
        "title": row["title"],
        "overview": row["overview"],
        "genres": row["genres"],
        "runtime": row["runtime"],
        "release_date": row["release_date"],
        "tmdb_rating": row["vote_average"],
        "imdb_rating": row["imdb_rating"],
        "director": row["director"],
        "cast": row["cast"],
        "poster": get_poster(row["poster_path"]),
    }

if __name__ == "__main__":
    movie = input("Enter Movie Name: ")
    recommendations = recommend(movie)
    if len(recommendations) == 0:
        print("\nMovie not found!")
    else:
        print("\nRecommended Movies:\n")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie['title']}")
            print(f"   Similarity : {movie['similarity']}%")
            print(f"   Hybrid Score : {movie['hybrid_score']}")
            print(f"   IMDb : {movie['imdb_rating']}")
            print(f"   TMDB : {movie['tmdb_rating']}")
            print()

