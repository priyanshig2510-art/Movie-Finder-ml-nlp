# Movie Finder

A Streamlit app that recommends movies using a content-based NLP model. Search for a film you like, or describe the kind of story you want, and get the closest matches from about 45,000 movies.

The model comes from the `NLP-01.ipynb` notebook: text cleaning with NLTK, TF-IDF vectors with scikit-learn, and cosine similarity.

## Features

- **Similar to a movie:** search by title and get recommendations based on synopsis, tagline and genres.
- **Describe what you want:** type a plot idea such as "heist crew plans the perfect robbery" and get the closest matches.
- **Featured top pick:** the best match gets a large card, and the rest appear in a poster grid with rating rings and match bars.
- **Filters:** number of results, minimum rating, required genres, and sort order (best match, highest rated, most popular).
- **Shortcuts:** a "Surprise me" button and one-click popular titles.
- **Optional real posters** loaded from TMDB (see [Adding posters](#adding-posters)).

## Project structure

```
.
├── app.py                  # Streamlit app
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark theme colours
├── NLP-01.ipynb            # Notebook that builds the model files
├── df.pkl                  # Created by the notebook
├── tfidf.pkl               # Created by the notebook
├── tfidf_matrix.pkl        # Created by the notebook
└── indices.pkl             # Created by the notebook (not used by the app)
```

## Setup

### 1. Install dependencies

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

Use the same scikit-learn version as the notebook, otherwise the pickle files may fail to load.

### 2. Build the model files

Run `NLP-01.ipynb` from top to bottom. The last cell saves four files:

```python
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
df.to_pickle('df.pkl')
pickle.dump(indices, open('indices.pkl', 'wb'))
```

The notebook downloads the [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle through `kagglehub`, so it needs an internet connection and may ask for Kaggle credentials.

Copy the generated `.pkl` files into the same folder as `app.py`.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

## How it works

1. **Cleaning:** each movie's `overview + genres + tagline` text is lowercased, stripped of punctuation and stop words, and lemmatized.
2. **Vectorizing:** `TfidfVectorizer` (5,000 features, unigrams and bigrams) turns every movie into a vector.
3. **Matching:** the app computes cosine similarity (`linear_kernel`) between the chosen movie, or your typed description, and every movie in the library.
4. **Filtering:** results are filtered by rating and genre, then the top matches are shown.

For free-text descriptions, your text goes through the same cleaning and the saved TF-IDF vectorizer before it is compared with the library.

## Adding posters

The notebook drops the `poster_path` column, so the app shows coloured tiles with a genre emoji. To show real posters:

1. In the notebook, keep the column when selecting columns:

   ```python
   df = df[['title', 'tagline', 'overview', 'genres', 'vote_average', 'popularity', 'poster_path']]
   ```

2. Re-run the notebook through the final pickle cell.
3. Replace `df.pkl` with the new file.

The app detects the column automatically. Posters load from `image.tmdb.org`, so an internet connection is required to see them.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "Model files not found" | Copy `df.pkl`, `tfidf.pkl` and `tfidf_matrix.pkl` next to `app.py`. |
| Error loading a pickle file | Install the same scikit-learn and pandas versions used in the notebook. |
| Description search returns nothing | Try different words. Only words that appear in the movie data can match. |
| Fonts look different | The Fraunces and Manrope fonts load from Google Fonts, so they need an internet connection. |
| Plain light theme | Make sure `.streamlit/config.toml` sits in a `.streamlit` folder next to `app.py`. |

## Possible improvements

- Add a space between `overview`, `genres` and `tagline` when building the `tag` column in the notebook. Currently the last word of one part merges with the first word of the next.
- Add release year and vote count to the data to filter out obscure titles with only a few ratings.

## Tech stack

Streamlit, pandas, NumPy, scikit-learn, NLTK.
