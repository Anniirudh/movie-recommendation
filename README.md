# 🎬 Movie Recommendation System

A content-based movie recommendation system built with **Python, Streamlit, and Scikit-learn**. The application helps users discover movies similar to their selected movie using movie metadata and similarity-based recommendations.

## ✨ Features

- 🎥 Movie recommendation based on content similarity
- 🔎 Interactive movie selection through a Streamlit interface
- 🤖 Machine learning using Scikit-learn
- 🐍 Python-based recommendation pipeline
- 🌐 Web interface for exploring recommendations

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Dataset:** TMDB movie dataset

## 📁 Project Structure

```text
movie-recommendation/
├── app.py
├── dataset/
│   └── TMDB_all_movies.csv  # Download separately
├── models/                  # If applicable
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure above is illustrative; adjust it to match the actual files in your repository.*

## 📊 Dataset

The application uses `TMDB_all_movies.csv`, a large movie dataset of approximately 739 MB.

The dataset is **not included in this GitHub repository** because it exceeds GitHub's standard 100 MB per-file limit.

To run the application:

1. Obtain the dataset from its original source, ensuring that you comply with its licensing and usage terms.
2. Create a `dataset` folder in the project root if it does not already exist.
3. Place the file at:

   `dataset/TMDB_all_movies.csv`

> The application may not work until the required dataset is available locally.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Anniirudh/movie-recommendation.git
cd movie-recommendation
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Download or obtain `TMDB_all_movies.csv` and place it inside the `dataset/` directory.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, typically `http://localhost:8501`.

## 🧠 How It Works

The application uses a content-based recommendation approach to identify movies similar to a user's selection.

In a typical content-based pipeline:

1. Movie metadata is loaded and processed.
2. Relevant movie features are represented numerically.
3. Similarity between movies is calculated.
4. The application returns movies with similar feature representations.

The precise features, vectorization method, and similarity metric depend on the implementation in the source code.

## ⚠️ Troubleshooting

- **Dataset not found:** Verify that `dataset/TMDB_all_movies.csv` exists at the expected path.
- **Missing Python packages:** Install dependencies using `python -m pip install -r requirements.txt`.
- **Scikit-learn version mismatch:** Use a compatible version of Scikit-learn for any serialized models created with a different version.
- **Streamlit launch issues:** Start the application with `python -m streamlit run app.py`.

## 🔮 Future Improvements

- Add movie posters and additional movie details.
- Improve recommendation quality through feature engineering.
- Add genre-based filters and search.
- Deploy the application online.
- Add automated tests and recommendation evaluation.

## 👨‍💻 Author

**Anniirudh**

GitHub: [@Anniirudh](https://github.com/Anniirudh)


