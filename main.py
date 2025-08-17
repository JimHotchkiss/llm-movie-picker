import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from util.helper import load_data
from util.function_calls import extract_genre_from_request, extract_VAD_from_request, extract_viewing_type_from_request, extract_rating_from_request
from util.data_filter import manually_filter_movies, extract_movie_vad_score

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

with st.sidebar:
    uploaded_file = st.file_uploader("Load Movie Data...", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = load_data(uploaded_file)
            if 'movie_dataframe' not in st.session_state:
                st.session_state['movie_dataframe'] = data
            st.write(data)
        elif uploaded_file.name.endswith(".xlsx"):
            data = load_data(uploaded_file)
            st.write(data)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
    
      
st.title("🍿 LLM Movie Picker")

# Initialize chat history
if "messages" not in st.session_state:
        st.session_state.messages = []
# Initialize movie criteria history
if "movie_criteria" not in st.session_state:
    st.session_state.movie_criteria = {
        "genre": None,
        "viewing_type": None,
        "rating": None,
        "VAD": {}
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
    if response.genre:
        set_genre_session(response.genre)
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
    with st.spinner("Extracting rating…"):
        rating_response = extract_rating_from_request({"query": user_query})
    if rating_response:
        set_rating_session(rating_response.rating)
    else:
        st.warning("No rating found in your query. Please try again.")
    # 4. Extract VAD from user query (if needed)
    with st.spinner("Extracting VAD…"):
        VAD_response = extract_VAD_from_request({"query": user_query})
    if VAD_response:
        set_vad_session(VAD_response)
        
    else:
        st.warning("No VAD found in your query. Please try again.")
    movie_vad_score = extract_movie_vad_score()
    filtered_df = manually_filter_movies()
    # set_filtered_data_sesseion(filtered_df)
    st.write(filtered_df)
   

def set_filtered_data_sesseion(df):
    st.session_state['filtered_df'] = df
    st.success(f"st.session_state['filtered_df'] set: {st.session_state['filtered_df']}")

def add_query_to_history(user_query: str):
    st.session_state.messages.append({"role": "user", "content": user_query})

def set_genre_session(response: list[str]):
    st.session_state.movie_criteria["genre"] = response
    st.success(f"Genre(s) extracted: {', '.join(response)}")

def set_view_type_session(response: list[str]):
    st.session_state.movie_criteria["viewing_type"] = response
    st.success(f"Viewing type(s) extracted: {', '.join(response)}")

def set_rating_session(response: str):
    st.session_state.movie_criteria["rating"] = response
    st.success(f"Rating extracted: {response}")

def set_vad_session(vad: dict):
    st.session_state.movie_criteria["VAD"] = {
        "valence":vad.vad.valence,
        "arousal": vad.vad.arousal,
        "dominance": vad.vad.dominance
    }
    st.success(f"VAD extracted: {vad.vad}")

# Respond to user input
if user_input := st.chat_input("What kind of movie are you in the mood for? Include genre, movie or TV and rating... 👋"):
    st.session_state.movie_criteria = {}
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    process_query(user_input)
    
 




