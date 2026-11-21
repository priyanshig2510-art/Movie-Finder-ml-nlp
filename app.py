"""Movie Finder: a Streamlit front end for the NLP-01 content-based recommender.

Expects these files (saved at the end of the notebook) next to this script:
    df.pkl, tfidf.pkl, tfidf_matrix.pkl
Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import pickle
import re
import textwrap
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Movie Finder", page_icon="🎬", layout="wide")

DATA_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = ["df.pkl", "tfidf.pkl", "tfidf_matrix.pkl"]

# TMDB genre names. The notebook joined genres with spaces, so multi-word
# names ("Science Fiction") are recovered by matching against this list.
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "Thriller", "TV Movie", "War",
    "Western",
]

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Manrope:wght@400;500;600;700&display=swap');

:root {
  --bg: #0F1722; --surface: #172233; --surface-2: #1D2A3F; --border: #2A3A52;
  --text: #E8ECF2; --muted: #8FA0B8; --gold: #E9B44C; --gold-dark: #1A1405;
}
.stApp { background: var(--bg); font-family: 'Manrope', sans-serif; color: var(--text); }
input, textarea, button { font-family: 'Manrope', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }

/* Hero with film-strip edges */
.hero {
  position: relative; overflow: hidden; border-radius: 22px; border: 1px solid var(--border);
  padding: 3.2rem 2.6rem 2.8rem; margin-bottom: 1.8rem;
  background:
    radial-gradient(900px 320px at 88% -30%, rgba(233,180,76,.26), transparent 60%),
    radial-gradient(700px 320px at -5% 130%, rgba(92,124,214,.26), transparent 60%),
    #131E2F;
}
.hero::before, .hero::after {
  content: ""; position: absolute; left: 0; right: 0; height: 10px;
  background: repeating-linear-gradient(90deg, transparent 0 14px, rgba(255,255,255,.11) 14px 26px);
}
.hero::before { top: 9px; } .hero::after { bottom: 9px; }
.hero h1 {
  font-family: 'Fraunces', serif; font-weight: 800; font-size: 3.6rem; line-height: 1.04;
  letter-spacing: -0.025em; margin: 0 0 .8rem 0; color: var(--text);
}
.hero p { color: #A9B7CC; font-size: 1.08rem; max-width: 36rem; line-height: 1.6; margin: 0; }
.hero .meta { display: inline-block; margin-top: 1.2rem; padding: .3rem .8rem; border-radius: 999px;
  background: rgba(233,180,76,.12); border: 1px solid rgba(233,180,76,.35); color: var(--gold); font-size: .82rem; font-weight: 600; }

/* Tabs as a segmented control */
.stTabs [data-baseweb="tab-list"] { background: var(--surface); padding: 5px; border-radius: 14px; gap: 4px; width: fit-content; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: auto; padding: .55rem 1.2rem; border-radius: 10px; color: var(--muted); font-weight: 700; background: transparent; }
.stTabs [aria-selected="true"] { background: var(--gold); color: var(--gold-dark); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; color: var(--text) !important; padding: .95rem 1.1rem !important; font-size: 1rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--gold) !important; box-shadow: 0 0 0 3px rgba(233,180,76,.18) !important; }
div[data-baseweb="select"] > div { background: var(--surface) !important; border-color: var(--border) !important; border-radius: 14px !important; }

/* Buttons */
.stButton > button {
  border-radius: 14px; border: 1px solid var(--border); background: var(--surface);
  color: var(--text); font-weight: 600; padding: .7rem 1rem; transition: all .15s;
}
.stButton > button:hover { border-color: var(--gold); color: var(--gold); transform: translateY(-1px); }
.stButton > button:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] h2 { font-family: 'Fraunces', serif; font-size: 1.35rem; }

/* Selected-movie banner */
.source { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--gold);
  border-radius: 14px; padding: 1.1rem 1.3rem; margin: 1.4rem 0 .2rem 0; }
.source .label { color: var(--muted); font-size: .85rem; margin-bottom: .2rem; }
.source .name { font-family: 'Fraunces', serif; font-size: 1.6rem; font-weight: 600; }
.source p { color: var(--muted); font-size: .92rem; line-height: 1.55; margin: .4rem 0 0 0; max-width: 60rem; }

.results-head { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 700; margin: 1.8rem 0 1rem 0; }

/* Posters */
.poster { position: relative; aspect-ratio: 2 / 3; border-radius: 12px; overflow: hidden;
  background-size: cover; background-position: center; box-shadow: 0 8px 24px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.08); }
.poster .fallback { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: .6rem; padding: 1rem; text-align: center; }
.poster .fallback .emoji { font-size: 2.6rem; }
.poster .fallback .ft { font-family: 'Fraunces', serif; font-weight: 600; font-size: .95rem; line-height: 1.2; color: rgba(255,255,255,.92); }
.rank { position: absolute; left: 8px; top: 8px; min-width: 26px; height: 26px; padding: 0 7px; border-radius: 8px;
  background: rgba(15,23,34,.85); color: var(--text); font-weight: 700; font-size: .8rem; display: grid; place-items: center; }
.ring { width: 42px; height: 42px; border-radius: 50%; display: grid; place-items: center;
  background: conic-gradient(var(--gold) var(--p), rgba(255,255,255,.2) 0); }
.ring span { width: 34px; height: 34px; border-radius: 50%; background: #0F1722; display: grid; place-items: center;
  font-size: .78rem; font-weight: 700; color: var(--text); }
.poster .ring { position: absolute; right: 8px; top: 8px; }

/* Grid cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: .7rem .7rem 1rem;
  margin-bottom: 1.1rem; transition: transform .18s, border-color .18s, box-shadow .18s; }
.card:hover { transform: translateY(-4px); border-color: var(--gold); box-shadow: 0 14px 30px rgba(0,0,0,.4); }
.card h3 { font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 600; line-height: 1.25; margin: .8rem .2rem .1rem;
  color: var(--text); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.6rem; }
.card .chips { margin: .4rem .2rem; min-height: 1.5rem; }
.card p { color: #AEBBD0; font-size: .82rem; line-height: 1.5; margin: .4rem .2rem .7rem;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; min-height: 3.7rem; }
.match { display: flex; align-items: center; gap: .6rem; font-size: .75rem; color: var(--muted); margin: 0 .2rem; }
.bar { flex: 1; height: 5px; background: var(--border); border-radius: 999px; overflow: hidden; }
.bar i { display: block; height: 100%; background: linear-gradient(90deg, #C98A1B, var(--gold)); border-radius: 999px; }

/* Featured top pick */
.feature { display: flex; gap: 1.8rem; padding: 1.4rem; border-radius: 20px; border: 1px solid var(--border); margin-bottom: 1.4rem;
  background: radial-gradient(600px 260px at 100% 0%, rgba(233,180,76,.14), transparent 65%), var(--surface); }
.feature .poster { flex: 0 0 200px; }
.feature .fbody { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center; }
.feature .flabel { color: var(--gold); font-weight: 700; font-size: .85rem; margin-bottom: .3rem; }
.feature h2 { font-family: 'Fraunces', serif; font-size: 2.2rem; font-weight: 700; line-height: 1.1; margin: 0; color: var(--text); letter-spacing: -.01em; }
.feature .tagline { margin-top: .4rem; font-size: .95rem; }
.feature p { color: #B7C3D6; font-size: .98rem; line-height: 1.65; margin: .4rem 0 1rem; max-width: 40rem; }
.feature .frow { display: flex; align-items: center; gap: 1rem; max-width: 26rem; }
.feature .frow .match { flex: 1; margin: 0; font-size: .82rem; }

.tagline { color: var(--muted); font-style: italic; font-size: .85rem; }
.chips { display: flex; flex-wrap: wrap; gap: .35rem; margin: .6rem 0; }
.chip { background: rgba(255,255,255,.05); border: 1px solid var(--border); color: #B7C3D6; border-radius: 999px; padding: .12rem .65rem; font-size: .74rem; font-weight: 600; }

.empty { color: var(--muted); padding: 1.6rem 0 .4rem 0; font-size: .98rem; }
.hint { color: var(--muted); font-size: .88rem; margin: 1.2rem 0 .5rem 0; }

@media (max-width: 700px) {
  .hero { padding: 2.4rem 1.4rem 2.2rem; } .hero h1 { font-size: 2.3rem; }
  .feature { flex-direction: column; } .feature .poster { flex-basis: auto; width: 55%; } .feature h2 { font-size: 1.6rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the movie library…")
def load_library():
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        return None, missing

    df = pd.read_pickle(DATA_DIR / "df.pkl").reset_index(drop=True)
    with open(DATA_DIR / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(DATA_DIR / "tfidf_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)

    for col in ("title", "tagline", "overview", "genres"):
        df[col] = df[col].fillna("").astype(str)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)

    if "poster_path" in df.columns:
        df["poster_path"] = df["poster_path"].fillna("").astype(str)
    else:
        df["poster_path"] = ""

    title_lower = df["title"].str.lower()
    first_index = pd.Series(np.arange(len(df)), index=df["title"])
    first_index = first_index[~first_index.index.duplicated()]

    lib = {
        "df": df,
        "tfidf": tfidf,
        "matrix": matrix,
        "title_lower": title_lower,
        "title_arr": df["title"].to_numpy(),
        "rating": df["vote_average"].to_numpy(),
        "popularity": df["popularity"].to_numpy(),
        "first_index": first_index,
        "genre_masks": {g: df["genres"].str.contains(g, regex=False).to_numpy() for g in GENRES},
    }
    return lib, []


@st.cache_resource
def get_preprocessor():
    """Same cleaning as the notebook (lowercase, strip punctuation, stop words,
    lemmatise). Falls back to light cleaning if NLTK data is unavailable."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        def _load():
            stop = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("tests")
            return stop, lemmatizer

        try:
            stop_words, lemmatizer = _load()
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            stop_words, lemmatizer = _load()

        def preprocess(text: str) -> str:
            text = re.sub(r"[^\w\s]", " ", text.lower())
            return " ".join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words)

        return preprocess
    except Exception:
        return lambda text: re.sub(r"[^\w\s]", " ", text.lower())


# --------------------------------------------------------------------------
# Search + recommendation logic
# --------------------------------------------------------------------------
def find_titles(lib, query: str, limit: int = 40) -> list[str]:
    """Titles matching the query: exact, then prefix, then substring, then all-words."""
    q = query.strip().lower()
    if not q:
        return []
    s = lib["title_lower"]
    exact = (s == q).to_numpy()
    starts = s.str.startswith(q).to_numpy()
    contains = s.str.contains(q, regex=False).to_numpy()
    words = np.ones(len(s), dtype=bool)
    for w in q.split():
        words &= s.str.contains(w, regex=False).to_numpy()

    mask = contains | words
    cand = np.flatnonzero(mask)
    if cand.size == 0:
        return []
    tier = np.where(exact, 0, np.where(starts, 1, np.where(contains, 2, 3)))[cand]
    order = np.lexsort((-lib["popularity"][cand], tier))
    titles = lib["title_arr"][cand[order]]
    return list(dict.fromkeys(titles))[:limit]


def top_matches(lib, scores, *, n, min_rating, genres, order_by, exclude_title=None):
    mask = (scores > 0) & (lib["rating"] >= min_rating)
    for g in genres:
        mask &= lib["genre_masks"][g]
    if exclude_title is not None:
        mask &= lib["title_arr"] != exclude_title
    cand = np.flatnonzero(mask)
    if cand.size == 0:
        return []
    top = cand[np.argsort(-scores[cand])[:n]]
    if order_by == "Highest rated":
        top = top[np.argsort(-lib["rating"][top], kind="stable")]
    elif order_by == "Most popular":
        top = top[np.argsort(-lib["popularity"][top], kind="stable")]
    return [(int(i), float(scores[i])) for i in top]


def genre_list(genre_string: str) -> list[str]:
    return [g for g in GENRES if g in genre_string]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🧭", "Animation": "🎨", "Comedy": "😄", "Crime": "🕵️",
    "Documentary": "🎞️", "Drama": "🎭", "Family": "👨‍👩‍👧", "Fantasy": "🧙", "Foreign": "🌍",
    "History": "🏛️", "Horror": "👻", "Music": "🎵", "Mystery": "🔍", "Romance": "💕",
    "Science Fiction": "🚀", "Thriller": "😰", "TV Movie": "📺", "War": "⚔️", "Western": "🤠",
}
POSTER_BASE = "https://image.tmdb.org/t/p/w342"


def poster_html(row, rank: int | None = None) -> str:
    """Poster image if the data has poster_path, otherwise a coloured tile."""
    title = row["title"]
    hue = zlib.crc32(title.encode("utf-8")) % 360
    gradient = f"linear-gradient(160deg, hsl({hue},48%,38%), hsl({(hue + 45) % 360},52%,16%))"
    path = row["poster_path"]
    if path.startswith("/"):
        style = f"background-image:url('{POSTER_BASE}{path}'), {gradient}"
        inner = ""
    else:
        genres = genre_list(row["genres"])
        emoji = GENRE_EMOJI.get(genres[0], "🎬") if genres else "🎬"
        style = f"background-image:{gradient}"
        inner = f'<div class="fallback"><div class="emoji">{emoji}</div><div class="ft">{html.escape(title)}</div></div>'
    rank_html = f'<div class="rank">{rank}</div>' if rank else ""
    rating = row["vote_average"]
    ring = f'<div class="ring" style="--p:{rating * 10:.0f}%"><span>{rating:.1f}</span></div>' if rating > 0 else ""
    return f'<div class="poster" style="{style}">{inner}{rank_html}{ring}</div>'


def chips_html(genre_string: str) -> str:
    return "".join(f'<span class="chip">{html.escape(g)}</span>' for g in genre_list(genre_string))


def match_html(score: float) -> str:
    pct = max(1, min(100, round(score * 100)))
    return f'<div class="match"><div class="bar"><i style="width:{pct}%"></i></div><span>{pct}% match</span></div>'


def card_html(row, rank: int, score: float) -> str:
    overview = textwrap.shorten(row["overview"], width=200, placeholder="…") or "No synopsis available."
    return (
        '<div class="card">'
        f"{poster_html(row, rank)}"
        f'<h3>{html.escape(row["title"])}</h3>'
        f'<div class="chips">{chips_html(row["genres"])}</div>'
        f"<p>{html.escape(overview)}</p>"
        f"{match_html(score)}"
        "</div>"
    )


def featured_html(row, score: float, label: str) -> str:
    overview = textwrap.shorten(row["overview"], width=420, placeholder="…") or "No synopsis available."
    tagline = f'<div class="tagline">{html.escape(row["tagline"])}</div>' if row["tagline"] else ""
    return (
        '<div class="feature">'
        f"{poster_html(row)}"
        '<div class="fbody">'
        f'<div class="flabel">{html.escape(label)}</div>'
        f'<h2>{html.escape(row["title"])}</h2>{tagline}'
        f'<div class="chips">{chips_html(row["genres"])}</div>'
        f"<p>{html.escape(overview)}</p>"
        f'<div class="frow">{match_html(score)}</div>'
        "</div></div>"
    )


def show_results(lib, results, order_by: str = "Best match"):
    if not results:
        st.markdown(
            '<div class="empty">No movies fit these filters. Lower the minimum rating or remove a genre in the sidebar.</div>',
            unsafe_allow_html=True,
        )
        return
    df = lib["df"]
    label = {"Best match": "Top pick", "Highest rated": "Highest rated", "Most popular": "Most popular"}[order_by]
    first_idx, first_score = results[0]
    st.markdown(featured_html(df.iloc[first_idx], first_score, label), unsafe_allow_html=True)

    rest = results[1:]
    for start in range(0, len(rest), 3):
        cols = st.columns(3, gap="medium")
        for col, (idx, score) in zip(cols, rest[start:start + 3]):
            rank = results.index((idx, score)) + 1
            with col:
                st.markdown(card_html(df.iloc[idx], rank, score), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
lib, missing = load_library()

if lib is None:
    st.markdown('<div class="hero"><h1>Movie Finder</h1></div>', unsafe_allow_html=True)
    st.error(
        "Model files not found: " + ", ".join(f"`{m}`" for m in missing)
        + f". Run the last cell of the notebook, then copy these files into `{DATA_DIR}`."
    )
    st.stop()

df = lib["df"]

# Sidebar filters
with st.sidebar:
    st.markdown("## Refine results")
    n_results = st.slider("Number of recommendations", 4, 20, 10, step=2)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, step=0.5)
    genre_filter = st.multiselect("Must include genres", GENRES)
    order_by = st.radio("Order results by", ["Best match", "Highest rated", "Most popular"])
    st.caption("Matches are based on each film's synopsis, tagline and genres.")

# Hero
st.markdown(
    f"""
    <div class="hero">
      <h1>Find your next movie</h1>
      <p>Name a film you love and get recommendations like it, or describe the kind of story you're in the mood for.</p>
      <div class="meta"><b>{len(df):,}</b> films in the library</div>
    </div>
    """,
    unsafe_allow_html=True,
)

popular_titles = list(
    dict.fromkeys(df.sort_values("popularity", ascending=False)["title"].head(60))
)


def set_state(key: str, value: str):
    st.session_state[key] = value


def surprise():
    pool = popular_titles[:40]
    st.session_state["movie_query"] = pool[np.random.randint(len(pool))]


tab_similar, tab_describe = st.tabs(["Similar to a movie", "Describe what you want"])

# ---- Tab 1: similar to a movie ------------------------------------------
with tab_similar:
    st.write("")
    c_input, c_btn = st.columns([5, 1.2], vertical_alignment="bottom")
    query = c_input.text_input(
        "Search for a movie",
        key="movie_query",
        placeholder="Search a title, e.g. Toy Story or Dark Knight",
    )
    c_btn.button("Surprise me", on_click=surprise, use_container_width=True)

    if not query.strip():
        st.markdown('<div class="hint">Popular picks to start with</div>', unsafe_allow_html=True)
        pick_cols = st.columns(5)
        for i, t in enumerate(popular_titles[:5]):
            pick_cols[i].button(t, key=f"pop_{i}", on_click=set_state, args=("movie_query", t), use_container_width=True)
    else:
        matches = find_titles(lib, query)
        if not matches:
            st.markdown(
                '<div class="empty">No title matches that search. Check the spelling, or describe the movie in the other tab.</div>',
                unsafe_allow_html=True,
            )
        else:
            chosen = matches[0]
            if len(matches) > 1:
                chosen = st.selectbox(f"{len(matches)} matching titles", matches)

            src_idx = int(lib["first_index"][chosen])
            src = df.iloc[src_idx]
            src_chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in genre_list(src["genres"]))
            src_overview = textwrap.shorten(src["overview"], width=300, placeholder="…")
            st.markdown(
                f'<div class="source"><div class="label">Recommendations based on</div>'
                f'<div class="name">{html.escape(chosen)}</div>'
                f'<div class="chips">{src_chips}</div>'
                f'<p>{html.escape(src_overview)}</p></div>',
                unsafe_allow_html=True,
            )

            scores = linear_kernel(lib["matrix"][src_idx], lib["matrix"]).ravel()
            results = top_matches(
                lib, scores, n=n_results, min_rating=min_rating,
                genres=genre_filter, order_by=order_by, exclude_title=chosen,
            )
            st.markdown('<div class="results-head">You might also like</div>', unsafe_allow_html=True)
            show_results(lib, results, order_by)

# ---- Tab 2: describe what you want --------------------------------------
with tab_describe:
    st.write("")
    desc = st.text_input(
        "Describe a movie",
        key="desc_query",
        placeholder="e.g. a detective hunts a serial killer in a rainy city",
    )

    if not desc.strip():
        st.markdown('<div class="hint">Try one of these</div>', unsafe_allow_html=True)
        examples = [
            "space crew stranded on a hostile planet",
            "heartwarming animated story about friendship",
            "heist crew plans the perfect robbery",
        ]
        ex_cols = st.columns(3)
        for i, ex in enumerate(examples):
            ex_cols[i].button(ex, key=f"ex_{i}", on_click=set_state, args=("desc_query", ex), use_container_width=True)
    else:
        cleaned = get_preprocessor()(desc)
        vec = lib["tfidf"].transform([cleaned])
        scores = linear_kernel(vec, lib["matrix"]).ravel()
        results = top_matches(
            lib, scores, n=n_results, min_rating=min_rating,
            genres=genre_filter, order_by=order_by,
        )
        if not results and scores.max() <= 0:
            st.markdown(
                '<div class="empty">None of those words appear in the library. Try describing the plot with different words.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="results-head">Closest matches</div>', unsafe_allow_html=True)
            show_results(lib, results, order_by)
            