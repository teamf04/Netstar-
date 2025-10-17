from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
from urllib.parse import unquote
from difflib import get_close_matches
import ast
import traceback

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# Load data
df = pd.read_csv('movies.csv')
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

def clean_dict(d):
    """Replace NaN with None, skip array ambiguity."""
    cleaned = {}
    for k, v in d.items():
        try:
            if isinstance(v, (list, tuple)) or hasattr(v, '__array__'):
                cleaned[k] = v  # skip pd.isna for arrays
            else:
                cleaned[k] = None if pd.isna(v) else v
        except Exception:
            cleaned[k] = v  # fallback: keep original
    return cleaned


def clean_df(df_slice):
    """Replace NaN with None in a DataFrame."""
    return df_slice.where(pd.notna(df_slice), None)

# Helper functions
def get_genre_recommendations(genre, n=20):
    mask = df['genre'].apply(lambda x: genre in x)
    return df[mask].sort_values('vote_average', ascending=False).head(n)

def get_director_movies(director):
    return df[df['director'] == director].sort_values('vote_average', ascending=False)

def get_cast_movies(cast_list):
    return df[df['cast'].apply(lambda x: any(item in cast_list for item in x))]

def get_recommendations(title):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]


def parse_genres(x):
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return val
        elif isinstance(val, str):
            return [val]
        else:
            return []
    except:
        if isinstance(x, str):
            return [x]
        else:
            return []

df['genre'] = df['genre'].apply(parse_genres)

# API Endpoints
@app.route('/genres')
def genres():
    all_genres = set()
    for genres_list in df['genre']:
        all_genres.update(genres_list)
    return jsonify(list(all_genres))

@app.route('/genre/<genre>')
def genre_recommendations(genre):
    results = get_genre_recommendations(genre)
    return jsonify(results[['title', 'vote_average', 'poster_url']].to_dict('records'))

@app.route('/movie/<title>')
def movie_details(title):
    try:
        title = unquote(title).strip().lower()
        df['title_lower'] = df['title'].str.lower().str.strip()

        match_row = df[df['title_lower'] == title]

        if match_row.empty:
            return jsonify({'error': 'Movie not found'}), 404

        movie = match_row.iloc[0]
        idx = match_row.index[0]

        if idx >= len(cosine_sim):
            return jsonify({'error': 'Movie index out of bounds'}), 500

        # Clean movie dictionary
        movie_dict = clean_dict(movie.to_dict())

        # Columns to return
        cols = ['title', 'poster_url']
        if 'url' in df.columns:
            cols.append('url')

        # Extract and clean cast
        cast_str = str(movie['cast']) if pd.notna(movie['cast']) else ''
        cast_members = [c.strip().lower() for c in cast_str.split(',') if isinstance(c, str) and c.strip()]

        # Similar movies by content
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        sim_indices = [i[0] for i in sim_scores if i[0] < len(df)]
        similar_by_content = clean_df(df.iloc[sim_indices][cols])

        # More by same director
        director_name = str(movie['director']).lower() if pd.notna(movie['director']) else ''
        director_movies = clean_df(
            df[df['director'].fillna('').astype(str).str.lower() == director_name][cols]
        )

        # Other movies by cast (final safe fix)
        cast_movies = df[cols].copy()
        cast_movies = cast_movies[
            df['cast']
            .fillna('')
            .astype(str)
            .apply(lambda cast: any(actor in cast.lower() for actor in cast_members if isinstance(actor, str)))
        ]
        cast_movies = clean_df(cast_movies)

        return jsonify({
            'movie': movie_dict,
            'similar_by_content': similar_by_content.to_dict(orient='records'),
            'similar_by_director': director_movies.to_dict(orient='records'),
            'similar_by_cast': cast_movies.to_dict(orient='records')
        })

    except Exception as e:
        print("Internal Server Error:", str(e))
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/movies')
def all_movies():
    try:
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
