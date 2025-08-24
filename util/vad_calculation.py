import numpy as np 
import pandas as pd

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
    m = _vec(movie_vad)
    w = np.asarray(w, dtype=float)

    # Weight the vad to de-emphasise the dominance
    u_w = u * w
    m_w = m * w

    # cosine
    nu = np.linalg.norm(u_w)
    nm = np.linalg.norm(m_w)
    cos = float((u_w @ m_w) / (nu * nm)) if nu > 0 and nm > 0 else 0.0

    # distance -> score in [0,1]
    d = float(np.linalg.norm(u_w - m_w))
    dmax = 2.0 * float(np.sqrt(np.sum(w)))  # max possible in weighted [-1,1]^3
    dist_score = 1.0 - (d / dmax)
    print(f"dist_score: {dist_score}")

    return alpha * cos + (1 - alpha) * dist_score


def rank_movies_by_vad(user_vad, movies_vad_array):
    print(f"movie_vad_array: {movies_vad_array}")
    # movies_vad_array: shape (N,3) in [0,1], same order as movie_ids
    scores = []
    for key in movies_vad_array:
        # 'valence': 0.9, 'arousal': 0.8, 'dominance'
        print(f"valence: {key['valence']}, arousal: {key['arousal']}, dominance: {key['dominance']}")
        # s = vad_similarity(user_vad, mv)
        # scores.append((mid, s))
    # sort high → low
    # return sorted(scores, key=lambda t: t[1], reverse=True)
    return 'Hi'