import streamlit as st
from recommender import (
    recommend,
    get_movie_details,
    search_movies,
)
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

# -----------------------------
# Load Movie Titles
# -----------------------------
movies = pickle.load(open("movies.pkl", "rb"))
movie_titles = sorted(movies["title"].tolist())

# -----------------------------
# Title
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.markdown(
    "Find movies similar to your favourites using AI-powered recommendations."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")

st.sidebar.info(
    """
This recommendation system uses

- TF-IDF
- Cosine Similarity
- Nearest Neighbors
- Hybrid Ranking
"""
)

# -----------------------------
# Movie Selection
# -----------------------------
selected_movie = st.selectbox(
    "Select a Movie",
    movie_titles,
)

# -----------------------------
# Show Movie Details
# -----------------------------
details = get_movie_details(selected_movie)

if details:

    col1, col2 = st.columns([1, 2])

    with col1:

        if details["poster"]:
            st.image(details["poster"], use_container_width=True)
        else:
            st.info("Poster not available")

    with col2:

        st.header(details["title"])

        st.write(f"**Genres:** {details['genres']}")
        st.write(f"**Director:** {details['director']}")
        st.write(f"**Runtime:** {details['runtime']} minutes")
        st.write(f"**Release Date:** {details['release_date']}")

        colA, colB = st.columns(2)

        with colA:
            st.metric("⭐ IMDb", details["imdb_rating"])

        with colB:
            st.metric("⭐ TMDB", details["tmdb_rating"])

        st.subheader("Overview")
        st.write(details["overview"])

        st.subheader("Cast")
        st.write(details["cast"])

st.divider()

# -----------------------------
# Recommend Button
# -----------------------------
if st.button("🎯 Recommend Similar Movies", use_container_width=True):

    with st.spinner("Finding similar movies..."):

        recommendations = recommend(selected_movie)

    st.success(f"Top {len(recommendations)} Recommendations")

    st.divider()

    for movie in recommendations:

        col1, col2 = st.columns([1, 3])

        with col1:

            if movie["poster"]:
                st.image(movie["poster"], use_container_width=True)
            else:
                st.info("No Poster")

        with col2:

            st.subheader(movie["title"])

            st.write(f"**Genres:** {movie['genres']}")
            st.write(f"**Director:** {movie['director']}")
            st.write(f"**Runtime:** {movie['runtime']} min")
            st.write(f"**Release Date:** {movie['release_date']}")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("IMDb", movie["imdb_rating"])

            with c2:
                st.metric("TMDB", movie["tmdb_rating"])

            with c3:
                st.metric(
                    "Similarity",
                    f"{movie['similarity']}%"
                )

            st.progress(movie["similarity"] / 100)

            st.write(movie["overview"])

        st.divider()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Made by Anirudh Prabhu")