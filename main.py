import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from util.helper import load_data
from util.genre_extracter import extract_genre_from_request
from util.function_calls import extract_VAD_from_request, extract_viewing_type_from_request, extract_audience_category_from_request, extract_movie_vad_score
from util.data_filter import manually_filter_movies
from util.vad_calculation import rank_movies_by_vad

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Ensure your API key is being correctly called
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
else:
    # (optionally mask most of it in logs)
    print(f"✅ OPENAI_API_KEY loaded: {openai_api_key[:4]}…")

client = OpenAI()
model="o4-mini"

st.set_page_config(page_title="AI Movie Picker")

with st.sidebar:
    uploaded_file = st.file_uploader("Load Movie Data...", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = load_data(uploaded_file)
            if 'movie_dataframe' not in st.session_state:
                st.session_state['movie_dataframe'] = data
            st.subheader(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            st.write(data)
        elif uploaded_file.name.endswith(".xlsx"):
            data = load_data(uploaded_file)
            st.write(data)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")

    st.markdown("""
    <style>
    .stButton > button {
        background-color: #0E1116; /* Green */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: #0E1116;
        color: grey
    }
    </style>
    """, unsafe_allow_html=True)

    st.button("Movie API Request")
# Content
st.subheader("🍿 LLM Movie Picker")
st.caption("Let’s find your perfect pic! Drop the genre, movie/TV, preferred rating, and a director if you’ve got one")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
        st.session_state.messages = []
# Initialize movie criteria history
if "movie_criteria" not in st.session_state:
    st.session_state.movie_criteria = {
        "llm_genre": {},
        "viewing_type": None,
        "audience_category": {},
        "VAD": {},
        "movie_VAD": []
    }
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = pd.DataFrame()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Define process functions 
def process_query(user_query: str):
    # 1. Add user query to session state
    add_query_to_history(user_query)
    # 2. Extract genre from user query
    with st.spinner("Extracting genres…"):
        response = extract_genre_from_request({"query": user_query})
    if response:
        set_genre_session(response)
    else:
        st.warning("No genre found in your query. Please try again.")
    # 3. Extract view type from user query (if needed)
    with st.spinner("Extracting viewing type…"):
        viewing_type_response = extract_viewing_type_from_request({"query": user_query})
    if viewing_type_response:
        set_view_type_session(viewing_type_response.viewing_type)
    else:
        st.warning("No viewing type found in your query. Please try again.")
    # 4. Extract rating from user query (if needed)
    with st.spinner("Extracting audience category…"):
        audience_category_response = extract_audience_category_from_request({"query": user_query})
    if audience_category_response.confidence < 0.6:
        st.warning(f"Unable to extract category, due to {audience_category_response.rationale}. Please, try again with this in mind")
        return 
    set_audience_category_session(audience_category_response)
    # 4. Extract VAD from user query (if needed)
    with st.spinner("Extracting VAD…"):
        VAD_response = extract_VAD_from_request({"query": user_query})
    if VAD_response:
        set_vad_session(VAD_response)
    else:
        st.warning("No VAD found in your query. Please try again.")
    # movie_vad_score = extract_movie_vad_score()
    return_filter_value = manually_filter_movies()
    if isinstance(return_filter_value, dict):
        st.warning(return_filter_value['message'])
        return 
    set_filtered_data_session(return_filter_value)
    filtered_movies_df_description = st.session_state['filtered_df'][['title','description']].head(3)
    movie_vad_score = extract_movie_vad_score()
    if movie_vad_score:
        user_vad = st.session_state.movie_criteria['VAD']
        movie_vad_array = st.session_state.movie_criteria['movie_VAD']
        return_movies_vad_ranking = rank_movies_by_vad(user_vad, movie_vad_array)
        print(f"st.session_state.movie_criteria['VAD']: {st.session_state.movie_criteria['VAD']}")
        st.write(movie_vad_score)
        st.write(return_movies_vad_ranking )  
        st.write(filtered_movies_df_description) 

def set_filtered_data_session(df):
    st.session_state['filtered_df'] = df
    st.success(f"Filtered dataframe set successfully: {st.session_state['filtered_df'].shape}")

def add_query_to_history(user_query: str):
    st.session_state.messages.append({"role": "user", "content": user_query})

def set_genre_session(response: dict):
    st.session_state.movie_criteria["llm_genre"] = {
        "genre": response.genre, 
        "confidence": response.confidence,
        "rationale": response.rationale
    }
    st.success(f"Genre extracted: {response}")

def set_view_type_session(response: list[str]):
    st.session_state.movie_criteria["viewing_type"] = response
    st.success(f"Viewing type(s) extracted: {', '.join(response)}")

def set_audience_category_session(response: str):
    st.session_state.movie_criteria["audience_category"] = {
        "category":response.category,
        "confidence": response.confidence,
        "rationale": response.rationale
    }
    st.success(f"Audience Category extracted: {response}")

def set_vad_session(vad: dict):
    st.session_state.movie_criteria["VAD"] = {
        "valence":vad.vad.valence,
        "arousal": vad.vad.arousal,
        "dominance": vad.vad.dominance
    }
    st.success(f"VAD extracted: {vad.vad}")

# Respond to user input
if user_input := st.chat_input("What kind of movie are you in the mood for? 👋"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    process_query(user_input)
    
 




