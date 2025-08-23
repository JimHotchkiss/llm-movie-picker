import os
import logging
import json
import streamlit as st
from models.models import ExtractDescription,ExtractVAD, ExtractMovieVAD, ExtractViewingType, ExtractAudienceCategory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
model = "o4-mini"

if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
else:
    # (optionally mask most of it in logs)
    print(f"✅ OPENAI_API_KEY loaded: {openai_api_key[:4]}…")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)



# 1) Few-shots: user/assistant pairs as STRINGS
view_type_few_shots = [
    {"role": "user", "content": "I want a TV show to binge this weekend."},
    {"role": "assistant", "content": json.dumps({"view_types": ["TV Series"]})},

    {"role": "user", "content": "Any good limited series about true crime?"},
    {"role": "assistant", "content": json.dumps({"view_types": ["Miniseries"]})},

    {"role": "user", "content": "Looking for a movie tonight."},
    {"role": "assistant", "content": json.dumps({"view_types": ["Movie"]})},

    {"role": "user", "content": "Either a movie or a show is fine."},
    {"role": "assistant", "content": json.dumps({"view_types": ["Movie", "TV Series"]})},

    # Your previous failing case, as a teaching example
    {"role": "user", "content": "I want to watch a crime fiction, TV series that has a strong romantic component."},
    {"role": "assistant", "content": json.dumps({"view_types": ["TV Series"]})},
]

SYSTEM_PROMPT = """
    You extract the desired viewing type. Output JSON only: {"view_types": ["Movie","TV Series","Miniseries"]} or {"view_types": []}.

    Rules:
    - Case-insensitive matching.
    - Normalize synonyms to exactly: "Movie", "TV Series", "Miniseries".
    - If multiple types are mentioned, list them in order of mention.
    - If both genres and type are present, still extract the type(s).
    - If ambiguous ("movie or show"), include both.
    - No extra keys or commentary.

    Synonyms (examples):
    - movie/film/feature → "Movie"
    - tv show/tv series/show/series/docuseries → "TV Series"
    - miniseries/mini-series/limited series/limited → "Miniseries"
""".strip()

def build_messages(request_text: str):
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + view_type_few_shots
        + [{"role": "user", "content": request_text}]  # <-- real input goes here
    )

def extract_viewing_type_from_request(request: dict) -> ExtractViewingType:
    logger.info(f"Extracting genre from request: {request}")
    message = build_messages(str(request))
    logger.info("Generating confirmation message")
    viewing_type_extraction = client.beta.chat.completions.parse(
        model=model,
        messages = message,
        response_format=ExtractViewingType
        )
    
    viewing_type_result = viewing_type_extraction.choices[0].message.parsed
    logger.info(f"Confirmation message generated successfully: {viewing_type_result}")
    return viewing_type_result

def build_audience_category_messages(request_text: str):
   cat_system_prompt = """
        You classify the intended audience category for a movie/TV request.

        Return JSON only in this exact schema:
        {"category":"<CHILDREN|TEEN|ADULT|>","confidence":<0..1>,"rationale":"<short reason>"}

        Guidelines:
        - Categories (coarse): CHILDREN ≈ under 12; TEEN ≈ 13–17; ADULT ≈ 18+.
        - Infer from age words and maturity cues (violence, language, sexual content, intensity). Don’t use mood alone.
        - If multiple cues conflict, choose the MOST RESTRICTIVE (ADULT > TEEN > CHILDREN).
        - Map ratings when present:
        G / TV-Y / TV-G / TV-Y7 → CHILDREN
        PG / TV-PG → TEEN
        PG-13 / TV-14 → TEEN
        R / NC-17 / TV-MA → ADULT
        - If unclear, return {"category":"","confidence":0.0,"rationale":"insufficient cues"}.
        - Keep rationale concise (<=12 words). Confidence to two decimals. No extra keys.
""".strip()
   
   cat_few_shots = [
       # CHILDREN
        {"role":"user","content":"Animated TV show for my 6-year-old—gentle, no scares."},
        {"role":"assistant","content": json.dumps({"category":"CHILDREN","confidence":0.95,"rationale":"explicitly for young child; gentle content"})},

        {"role":"user","content":"Wholesome family movie everyone can enjoy—no crude jokes."},
        {"role":"assistant","content": json.dumps({"category":"CHILDREN","confidence":0.90,"rationale":"family-friendly all ages cues"})},

        {"role":"user","content":"Nature documentary film that’s kid-safe and educational."},
        {"role":"assistant","content": json.dumps({"category":"CHILDREN","confidence":0.88,"rationale":"kid-safe educational; no mature content"})},

        # TEEN
        {"role":"user","content":"PG family adventure—some peril, nothing heavy."},
        {"role":"assistant","content": json.dumps({"category":"TEEN","confidence":0.80,"rationale":"PG mapping; mild peril"})},

        {"role":"user","content":"Teen-friendly sci-fi series—some language, no explicit content."},
        {"role":"assistant","content": json.dumps({"category":"TEEN","confidence":0.82,"rationale":"teen-friendly with mild language"})},

        {"role":"user","content":"Mystery/thriller movie suitable for older teens—tense but not graphic."},
        {"role":"assistant","content": json.dumps({"category":"TEEN","confidence":0.84,"rationale":"older teen cues; non-graphic"})},

        # ADULT
        {"role":"user","content":"Dark crime drama miniseries for adults—strong language and sex scenes."},
        {"role":"assistant","content": json.dumps({"category":"ADULT","confidence":0.94,"rationale":"adult themes: strong language and sexual content"})},

        {"role":"user","content":"R-rated horror movie—graphic violence and disturbing imagery."},
        {"role":"assistant","content": json.dumps({"category":"ADULT","confidence":0.97,"rationale":"R-rated; graphic violence cues"})},

        {"role":"user","content":"Adult animated comedy with explicit language and raunchy humor."},
        {"role":"assistant","content": json.dumps({"category":"ADULT","confidence":0.92,"rationale":"explicit language and adult humor"})},

        # Most-restrictive rule
        {"role":"user","content":"Mostly family-friendly but occasional strong profanity."},
        {"role":"assistant","content": json.dumps({"category":"TEEN","confidence":0.72,"rationale":"family-friendly with strong language; choose stricter"})},

        # Ambiguous / insufficient
        {"role":"user","content":"Surprise me with a good sci-fi series."},
        {"role":"assistant","content": json.dumps({"category":"","confidence":0.0,"rationale":"insufficient cues"})},
   ]

   return [{"role": "system", "content": cat_system_prompt}] + cat_few_shots + [
        {"role": "user", "content": request_text}
    ]
  
def extract_audience_category_from_request(request: dict) -> ExtractAudienceCategory:
    logger.info(f"Extracting rating from request: {request}")

    message = build_audience_category_messages(str(request))

    rating_extraction = client.beta.chat.completions.parse(
        model=model,
        messages = message,
        response_format=ExtractAudienceCategory
        )
    
    msg = rating_extraction.choices[0].message
    llm_raw_response = msg.content

    logger.debug(f"RAW assistant: \n%s {llm_raw_response}")
    
    rating_result = rating_extraction.choices[0].message.parsed
    logger.info(f"Confirmation message generated successfully: {rating_result}")
    return rating_result

def extract_description_from_request(request: dict) -> ExtractDescription:
    logger.info(f"Extracting description from request: {request}")
    """
    Extracts the description from the request dictionary and returns an ExtractDescription model instance.

    Args:
        request (dict): The request dictionary containing the VAD information.
    Returns:
        ExtractVAD: An instance of ExtractVAD with the extracted VAD.
    """
    logger.info("Generating confirmation message")

    description_extraction = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are a movie or TV show description extraction assistant.

                Your task:
                - Identify and extract the movie or TV show description mentioned in the user’s request.
                - Return ONLY the description as a string in the exact wording the user provided.
                - If no description is found, return an empty string.

                Valid descriptions include but are not limited to:
                Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, War, Western.

                Rules:
                1. Do not include extra commentary or unrelated text.
                2. Genres must be capitalized properly (e.g., "Science Fiction" → "Sci-Fi").
                3. Multiple genres should be listed in the order they are mentioned.
                4. If a subgenre is mentioned (e.g., "Romantic Comedy"), include it as is.
                5. If the genre is vague (e.g., "scary movies"), map it to the closest valid genre (e.g., "Horror").

                Examples:

                User: "Something light and feel-good about friends in their 30s figuring out life in a big city."
                Output: "Something light and feel-good about friends in their 30s figuring out life in a big city."

                User: "A slow-burn thriller set in snowy mountains with a missing person case and an unreliable narrator."
                Output: "A slow-burn thriller set in snowy mountains with a missing person case and an unreliable narrator."

                User: "Action and comedy." 
                Output: ""

                User: "Family-friendly fantasy with gentle humor and a cozy, magical small town."
                Output: "Family-friendly fantasy with gentle humor and a cozy, magical small town."

                User: "A true-crime docuseries that focuses more on the investigation details than gore."
                Output: "A true-crime docuseries that focuses more on the investigation details than gore."

                User: "Rom-com but not too cheesy; set in Europe; diverse cast; under 2 hours."
                Output: "Rom-com but not too cheesy; set in Europe; diverse cast; under 2 hours."

                ---
                Return only the description string.
                """
            },
            {
                "role": "user",
                "content": str(request)  # Or pass the user's request string directly
            }
            ],
            response_format=ExtractDescription,
        )
    description_result = description_extraction.choices[0].message.parsed
    logger.info(f"Confirmation message generated successfully: {description_result}")
    return description_result

def extract_VAD_from_request(request: dict) -> ExtractVAD:
    logger.info(f"Extracting VAD from request: {request}")
    """
   You estimate Valence–Arousal–Dominance (VAD) for text.

    Output JSON only:
    {"vad":{"valence":0.0-1.0,"arousal":0.0-1.0,"dominance":0.0-1.0},"rationale":"<short>"}

    Guidelines:
    - Valence: unpleasant/sad=0.0 → pleasant/joyful=1.0
    - Arousal: calm/slow=0.0 → intense/exciting=1.0
    - Dominance: powerless/hemmed-in=0.0 → in-control/empowered=1.0
    - Be concise. No extra keys.

    Examples:
    Text: "Cozy, low-stakes story about friends helping each other."
    Output: {"vad":{"valence":0.8,"arousal":0.3,"dominance":0.5},"rationale":"Warm, calm support."}

    Text: "Bleak slow-burn mystery that feels suffocating."
    Output: {"vad":{"valence":0.2,"arousal":0.55,"dominance":0.3},"rationale":"Bleak tone, pressure."}
    """
    logger.info("Generating confirmation message")

    description_extraction = client.beta.chat.completions.parse(
        model=model,
        seed=7,
        messages = [
            {
                "role": "system",
                "content": (
                    "You estimate Valence–Arousal–Dominance (VAD) for a short text about movie/TV preferences or synopses. "
                    "Definitions: Valence (unpleasant→pleasant), Arousal (calm→intense), Dominance (powerless→in-control). "
                    "All values must be in [0,1]. Return only what the response schema expects."
                ),
            },
            # Few-shots (user → assistant). The assistant replies match your schema but you can keep them concise:
            {
                "role": "user",
                "content": "Something light and cozy about friendship, low stakes, gentle humor."
            },
            {"role": "assistant", "content": '{"vad":{"valence":0.80,"arousal":0.30,"dominance":0.50},"rationale":"Warm, calm, supportive vibe."}'},
            {
                "role": "user",
                "content": "Bleak, slow-burn mystery that feels suffocating."
            },
            {"role": "assistant", "content": '{"vad":{"valence":0.20,"arousal":0.55,"dominance":0.30},"rationale":"Low mood, mid arousal, low control."}'},
            # Real input
            {"role": "user", "content": str(request)},
        ],
            response_format=ExtractVAD,
        )
    VAD_result = description_extraction.choices[0].message.parsed
    logger.info(f"Confirmation message generated successfully: {VAD_result}")
    return VAD_result


MOVIE_VAD_SYSTEM_PROMPT = """

   You estimate Valence–Arousal–Dominance (VAD) for text.

    Output JSON only:
    {"vad":{"valence":0.0-1.0,"arousal":0.0-1.0,"dominance":0.0-1.0},"rationale":"<short>"}

    Guidelines:
    - Valence: unpleasant/sad=0.0 → pleasant/joyful=1.0
    - Arousal: calm/slow=0.0 → intense/exciting=1.0
    - Dominance: powerless/hemmed-in=0.0 → in-control/empowered=1.0
    - Be concise. No extra keys.

    Examples:
    Text: "Cozy, low-stakes story about friends helping each other."
    Output: {"vad":{"valence":0.8,"arousal":0.3,"dominance":0.5},"rationale":"Warm, calm support."}

    Text: "Bleak slow-burn mystery that feels suffocating."
    Output: {"vad":{"valence":0.2,"arousal":0.55,"dominance":0.3},"rationale":"Bleak tone, pressure."}
"""

movie_vad_few_shots = [
    # Few-shots (user → assistant). The assistant replies match your schema but you can keep them concise:
            {"role": "user","content": "Something light and cozy about friendship, low stakes, gentle humor."},
            {"role": "assistant", "content": '{"vad":{"valence":0.80,"arousal":0.30,"dominance":0.50},"rationale":"Warm, calm, supportive vibe."}'},
            {"role": "user", "content": "Bleak, slow-burn mystery that feels suffocating."},
            {"role": "assistant", "content": '{"vad":{"valence":0.20,"arousal":0.55,"dominance":0.30},"rationale":"Low mood, mid arousal, low control."}'},         
]

def build_movie_vad_prompt(movie_description: str):
     return (
        [{"role": "system", "content": MOVIE_VAD_SYSTEM_PROMPT}]
        + movie_vad_few_shots
        + [{"role": "user", "content": movie_description}]  # <-- real input goes here
    )

def extract_movie_vad_score() -> ExtractMovieVAD:
    filtered_df = st.session_state['filtered_df'].iloc[:3]
    movie_vad_score_obj = {}
    for index, row in filtered_df.iterrows():
        movie_description = row['description']
        movie_id = row['show_id']
        movie_vad_score = client.beta.chat.completions.parse(
            model=model,
            seed=7,
            messages=build_movie_vad_prompt(movie_description),
            response_format=ExtractMovieVAD,
        )
        movie_VAD_result = movie_vad_score.choices[0].message.parsed
        logger.info(f"Confirmation message generated successfully: {movie_VAD_result}")
        if movie_VAD_result:
            print(f"movie_VAD_result: {movie_VAD_result.movie_vad}")
            print(f"st.session_state.movie_criteria:{ st.session_state.movie_criteria}")
            movie_vad_score_obj = {
                "valence": movie_VAD_result.movie_vad.valence,
                "arousal": movie_VAD_result.movie_vad.arousal,
                "dominance": movie_VAD_result.movie_vad.dominance,
                "movie_id": movie_id
            }
            st.session_state.movie_criteria["movie_VAD"].append(movie_vad_score_obj)
    movie_VAD_lst = st.session_state.movie_criteria["movie_VAD"]
    return movie_VAD_lst
    
