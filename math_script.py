import numpy as np 
import streamlit as st
import pandas as pd

def angle_between_vectors(v1, v2):
    """
    Calculates the angle between two vectors in radians.

    Args:
        v1 (numpy.ndarray or list): The first vector.
        v2 (numpy.ndarray or list): The second vector.

    Returns:
        float: The angle between the vectors in radians.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)


    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    
    cosine_angle = dot_product/(magnitude_v1 * magnitude_v2)

    # Clip values to ensure they are within the valid range arcos (-1, 1)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_radians = np.arccos(cosine_angle)

    return angle_radians

vector_a = [0.85, 0.35, 0.6]
vector_b = [0.2, 0.8, 0.3]

# valence=0.85 arousal=0.35 dominance=0.6
# valence=0.2 arousal=0.8 dominance=0.3

# angle_rad = angle_between_vectors(vector_a, vector_b)
# angle_deg = np.degrees(angle_rad)

# print(f"Angle between vectors (radians): {angle_rad}")
# print(f"Angle between vectors (degrees): {angle_deg}")




def _vec(vad: dict | tuple | list) -> np.ndarray:
    # expects {'valence': x, 'arousal': y, 'dominance': z} or [x,y,z]
    if isinstance(vad, dict):
        x = [vad["valence"], vad["arousal"], vad["dominance"]]
    else:
        x = vad
    a = np.asarray(x, dtype=float)
    # clamp to [0,1] just in case, center to [-1,1]
    a = np.clip(a, 0.0, 1.0) * 2.0 - 1.0
    return a

def vad_similarity(user_vad, movie_vad, w=(1.0, 1.0, 0.7), alpha=0.6) -> float:
    u = _vec(user_vad)
    m1 = _vec(movie_vad[0])
    m2 = _vec(movie_vad[1])
    m3 = _vec(movie_vad[2])
    w = np.asarray(w, dtype=float)
    print(f"m1, m2, m3: {m1}, {m2},{m2}")
    return "HI"

    # # Weight the vad to de-emphasise the dominance
    # u_w = u * w
    # m_w = m * w

    # # cosine
    # nu = np.linalg.norm(u_w)
    # nm = np.linalg.norm(m_w)
    # cos = float((u_w @ m_w) / (nu * nm)) if nu > 0 and nm > 0 else 0.0

    # # distance -> score in [0,1]
    # d = float(np.linalg.norm(u_w - m_w))
    # dmax = 2.0 * float(np.sqrt(np.sum(w)))  # max possible in weighted [-1,1]^3
    # dist_score = 1.0 - (d / dmax)
    # print(f"dist_score: {dist_score}")

    # return alpha * cos + (1 - alpha) * dist_score

def rank_movies_by_vad(user_vad, movies_vad_array):
    # movies_vad_array: shape (N,3) in [0,1], same order as movie_ids
    scores = []
    for mv, mid in zip(movies_vad_array):
        s = vad_similarity(user_vad, mv)
        scores.append((mid, s))
    # sort high → low
    return sorted(scores, key=lambda t: t[1], reverse=True)

user_vad = vector_a
movie_vad = vector_b

vad_similarity(user_vad, movie_vad)

def split_unique_genre() -> list:
    unique_genres = []
    movie_df = pd.read_csv('./data/netflix_titles.csv')
    movie_df_renamed = movie_df.rename(columns={"listed_in": "genre"})
    for column in movie_df_renamed['genre']:
        str_to_list = column.split()
        for word in str_to_list:
            if word not in unique_genres:
                unique_genres.append(word)
    print(f"unique_genres: {unique_genres}")
    return unique_genres

split_unique_genre()


