import pandas as pd
import streamlit as st

def manually_filter_movies() -> pd.DataFrame:
    movie_df = st.session_state['movie_dataframe']
    llm_genre = st.session_state['movie_criteria']['llm_genre']['genre']
    genre_obj = st.session_state['movie_criteria']['llm_genre']
    viewing_type = st.session_state['movie_criteria']['viewing_type']
    if llm_genre:
        genre_filtered_movie_df = movie_df[movie_df['genre'].str.contains(llm_genre)]
        if genre_filtered_movie_df.empty:
            return {"message": f"Unable to filter on {llm_genre}. Please modify your request with this in mind"}
    genre_view_type_filtered_movie_df = filter_view_type(genre_filtered_movie_df)
    if genre_view_type_filtered_movie_df.empty:
        return {"message": f"Unable to filter on {viewing_type}. Please modify your request with this in mind"}
    genre_view_type_audience_category_movie_df = filter_audience_category(genre_view_type_filtered_movie_df)
    print(f"genre_view_type_audience_category_movie_df: {genre_view_type_audience_category_movie_df}")
    if genre_view_type_audience_category_movie_df.empty:
        return {"message": f"Unable to filter on {viewing_type}. Please modify your request with this in mind"}
    return genre_view_type_audience_category_movie_df

def filter_view_type(genre_filtered_movie_df: pd.DataFrame) -> pd.DataFrame:
    view_type_lst = st.session_state['movie_criteria']['viewing_type']
    new_view_type_lst = split_list(view_type_lst)
    if len(new_view_type_lst) > 1:
        view_type_movie_df = genre_filtered_movie_df[genre_filtered_movie_df['type'].apply(lambda lst: any(item in lst for item in new_view_type_lst))]
        return view_type_movie_df
    # It's not capturing 'Mystery' because in the df it's listed as 'Mysteries'
    view_type_movie_df = genre_filtered_movie_df[genre_filtered_movie_df['type'].apply(lambda lst: any(item in lst for item in view_type_lst))]
    return view_type_movie_df

def filter_audience_category(df):
    print(f"st.session_state.movie_criteria['audience_category']: {st.session_state.movie_criteria['viewing_type'][0]}")
    if st.session_state.movie_criteria['viewing_type'][0] == "Movie":
        if st.session_state.movie_criteria["audience_category"]:
            audience_category = st.session_state.movie_criteria["audience_category"]["category"]
        if audience_category == "CHILDREN":
            print(f"CHILDREN - audience_category: {audience_category}")
            rating = "G"
            audience_category_df  = df[df['rating'] == rating]
            return audience_category_df
        if audience_category == "TEEN":
            print(f"TEEN - audience_category: {audience_category}")
            rating = "PG-13"
            audience_category_df  = df[df['rating'] == rating]
            return audience_category_df
        if audience_category == "ADULT":
            print(f"ADULT - audience_category: {audience_category}")
            rating = "R"
            audience_category_df  = df[df['rating'] == rating]
            return audience_category_df
    if st.session_state.movie_criteria["audience_category"]:
            audience_category = st.session_state.movie_criteria["audience_category"]["category"]
            if audience_category == "CHILDREN":
                rating = "TV-Y7"
                audience_category_df  = df[df['rating'] == rating ]
                return audience_category_df
            if audience_category == "TEEN":
                rating = "TV-14"
                audience_category_df  = df[df['rating'] == rating]
                return audience_category_df
            if audience_category == "ADULT":
                rating = "TV-MA"
                audience_category_df  = df[df['rating'] == rating]
                return audience_category_df
def split_list(lst: list[str]) -> list[str]:
    output_lst: list[str] = []
    for element in lst:
        element = element.strip() # Removes leading and trailing white spaces
        if " " in element:
            output_lst.extend([word for word in element.split() if word]) # Split the string into words, ignore any empty ones, and add all the words to the output list.
        else:
            output_lst.append(element)
    return output_lst




