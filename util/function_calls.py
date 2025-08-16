import os
import logging
import json
from models.models import ExtractGenre, ExtractDescription,     ExtractVAD, ExtractViewingType
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



def extract_genre_from_request(request: dict) -> ExtractGenre:
    logger.info(f"Extracting genre from request: {request}")
    """
    Extracts the genre from the request dictionary and returns an ExtractGenre model instance.

    Args:
        request (dict): The request dictionary containing the genre information.
    Returns:
        ExtractGenre: An instance of ExtractGenre with the extracted genre.
    """
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    genre_extraction = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are a movie genre extraction assistant.

                Your task:
                - Identify and extract the movie genre or genres mentioned in the user’s request.
                - Return ONLY the genres as a list of strings in the exact wording the user provided (or the closest standard movie genre equivalent).
                - If no genre is found, return an empty list.

                Valid genres include but are not limited to:
                Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, War, Western.

                Rules:
                1. Do not include extra commentary or unrelated text.
                2. Genres must be capitalized properly (e.g., "Science Fiction" → "Sci-Fi").
                3. Multiple genres should be listed in the order they are mentioned.
                4. If a subgenre is mentioned (e.g., "Romantic Comedy"), include it as is.
                5. If the genre is vague (e.g., "scary movies"), map it to the closest valid genre (e.g., "Horror").

                Examples:

                User: "I want to watch a romantic comedy with friends."
                Output: ["Romantic Comedy"]

                User: "Find me some action and adventure films."
                Output: ["Action", "Adventure"]

                User: "Can you give me a good documentary?"
                Output: ["Documentary"]

                User: "Surprise me with any movie."
                Output: []

                User: "I'd like horror, mystery, or thriller."
                Output: ["Horror", "Mystery", "Thriller"]

                User: "I'd like an international movie with romance and comedy."
                Output: ["International", "Romance", "Comedy"]

                User: "What’s a good family-friendly fantasy?"
                Output: ["Family", "Fantasy"]

                ---
                Return only the extracted genres as JSON in this structure:
                {"genre": ["<Genre1>", "<Genre2>", ...]}
                """
            },
            {
                "role": "user",
                "content": str(request)  # Or pass the user's request string directly
            }
            ],
            response_format=ExtractGenre,
        )
    genre_result = genre_extraction.choices[0].message.parsed
    logger.info(f"Confirmation message generated successfully: {genre_result}")
    return genre_result

# 1) Few-shots: user/assistant pairs as STRINGS
few_shots = [
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

logger.info(f"few_shot: {few_shots}")

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
        + few_shots
        + [{"role": "user", "content": request_text}]  # <-- real input goes here
    )

def extract_viewing_type_from_request(request: dict) -> ExtractViewingType:
    logger.info(f"Extracting genre from request: {request}")
    """
    Extract the user's desired **view type** (e.g., Movie, TV Series, Miniseries) from a free-text request
    and return it as an `ExtractViewType` model instance.

    This function inspects the request text, normalizes common synonyms, and selects a single canonical
    label. It is case-insensitive and ignores unrelated content (genres, titles, etc.). If no clear view
    type is detected, the model will contain an empty/None value per the `ExtractViewType` schema.

    Canonical values (normalized):
    - "Movie"        (e.g., "movie", "film", "feature")
    - "TV Series"    (e.g., "tv show", "show", "series")
    - "Miniseries"   (e.g., "mini-series", "limited series", "limited")

    Args:
        request (dict): A request payload containing the user's text (for example, under "query" or
            similar key). The function will read the relevant text field(s) to infer the view type.

    Returns:
        ExtractViewType: A Pydantic model instance with the extracted and normalized view type.
            If no view type is found, the instance will reflect an empty/None value as defined
            by your model.

    Notes:
    - When multiple view types are mentioned, the first explicit, unambiguous mention is preferred.
    - Synonyms and abbreviations are mapped to the canonical values listed above.
    - Ambiguous phrasing (e.g., "something to binge") does not force a guess; the function returns
    empty/None unless a view type is clearly indicated.

    Raises:
        pydantic.ValidationError: If the constructed payload does not conform to `ExtractViewingType`.

    Examples:
        extract_viewing_type_from_request({"query": "I want a TV show to binge"})
        ExtractViewingType(view_type="TV Series")

        extract_viewing_type_from_request({"query": "Any good limited series?"})
        ExtractViewingType(view_type="Miniseries")

        extract_viewing_type_from_request({"query": "Surprise me"})
        ExtractViewingType(view_type=None)  # or "", depending on your schema
    """
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