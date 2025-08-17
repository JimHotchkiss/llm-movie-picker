import pandas as pd
import streamlit as st

def manually_filter_movies() -> pd.DataFrame:
    movie_df = st.session_state['movie_dataframe']

    genre_lst = st.session_state['movie_criteria']['genre']
    new_genre_lst = split_list(genre_lst)
    if len(new_genre_lst) > 1:
        genre_filtered_movie_df = movie_df[movie_df['genre'].apply(lambda lst: any(item in lst for item in new_genre_lst))]
        print(f"line 12. genre_filtered_movie_df: {genre_filtered_movie_df}")
    else:
        genre_filtered_movie_df = movie_df[movie_df['genre'].str.contains(genre_lst[0], case=False, na=False)]
        # genre_filtered_movie_df = movie_df[movie_df['genre'].apply(lambda lst: any(item in lst for item in genre_lst))]
        print(f"genre_filtered_movie_df: {genre_filtered_movie_df}")
        genre_view_type_filtered_movie_df = filter_view_type(genre_filtered_movie_df)
        print(f"genre_view_type_filtered_movie_df: {genre_filtered_movie_df}")

    genre_view_type_filtered_movie_df = filter_view_type(genre_filtered_movie_df)
    print(f"line 22. genre_view_type_filtered_movie_df: {genre_filtered_movie_df}")
    return genre_view_type_filtered_movie_df

def filter_view_type(genre_filtered_movie_df: pd.DataFrame) -> pd.DataFrame:
    view_type_lst = st.session_state['movie_criteria']['viewing_type']
    new_view_type_lst = split_list(view_type_lst)
    if len(new_view_type_lst) > 1:
        view_type_movie_df = genre_filtered_movie_df[genre_filtered_movie_df['type'].apply(lambda lst: any(item in lst for item in new_view_type_lst))]
        return view_type_movie_df
    # It's not capturing 'Mystery' because in the df it's listed as 'Mysteries'
    view_type_movie_df = genre_filtered_movie_df[genre_filtered_movie_df['type'].apply(lambda lst: any(item in lst for item in view_type_lst))]
    return view_type_movie_df

def split_list(lst: list[str]) -> list[str]:
    output_lst: list[str] = []
    for element in lst:
        element = element.strip() # Removes leading and trailing white spaces
        if " " in element:
            output_lst.extend([word for word in element.split() if word]) # Split the string into words, ignore any empty ones, and add all the words to the output list.
        else:
            output_lst.append(element)
    return output_lst


def extract_movie_vad_score() -> pd.DataFrame:
    filtered_df = st.session_state['filtered_df']
    for index, row in filtered_df.iterrows():
        print(f"index: {index}")
        print(f"row: {row}")
    return 'hi'
