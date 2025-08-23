import os
import logging
import json
from models.models import ExtractGenre, ExtractDescription,ExtractVAD, ExtractViewingType, ExtractAudienceCategory
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

ALLOWED_GENRES = [
    "Documentaries", "Docuseries", "Reality TV",
    "International TV Shows", "British TV Shows", "Korean TV Shows",
    "Kids' TV", "Teen TV Shows", "Talk Shows", "Sports Series", "Science & Nature TV",
    "TV Dramas", "TV Mysteries", "TV Comedies", "TV Action & Adventure",
    "Children & Family Movies", "Independent Movies", "Classic Movies", "Anime Features",
    "LGBTQ Movies", "Horror Movies", "Thrillers", "Comedy", "Music & Musicals",
    "Cult Movies", "Spanish-Language", "Faith & Spirituality", "Sci-Fi & Fantasy"
]

genre_few_shots = [
    # Action vs Crime → set-pieces dominate → Action label
    {"role": "user", "content": "Gritty underworld vibe but I want wild car chases and shootouts."},
    {"role": "assistant", "content": json.dumps({
        "genre": "TV Action & Adventure",
        "confidence": 0.82,
        "rationale": "Emphasis on chases and firefights signals an action-driven experience."
    })},

    # Crime/Thriller investigation focus
    {"role": "user", "content": "Detectives unravel a mafia conspiracy through patient investigation."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Thrillers",
        "confidence": 0.86,
        "rationale": "Tension and criminal intrigue align with thriller conventions."
    })},

    # Horror vs Thriller → sustained fear
    {"role": "user", "content": "An unseen presence stalks a group in an isolated cabin all night."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Horror Movies",
        "confidence": 0.9,
        "rationale": "Sustained fear and predatory threat define horror."
    })},

    # Non-fiction decision
    {"role": "user", "content": "In-depth real-world whistleblower story with interviews and archival footage."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Documentaries",
        "confidence": 0.92,
        "rationale": "Explicitly non-fiction presentation."
    })},

    # Family film preference
    {"role": "user", "content": "Something uplifting the whole family can watch together this weekend."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Children & Family Movies",
        "confidence": 0.88,
        "rationale": "Family-friendly movie preference is explicit."
    })},

    # International bingeable series
    {"role": "user", "content": "A bingeable international series about relationships and life transitions."},
    {"role": "assistant", "content": json.dumps({
        "genre": "International TV Shows",
        "confidence": 0.84,
        "rationale": "Bingeable series with international focus fits this label."
    })},

    # Regional TV variant
    {"role": "user", "content": "I’m into dry humor from the UK—recommend a series."},
    {"role": "assistant", "content": json.dumps({
        "genre": "British TV Shows",
        "confidence": 0.85,
        "rationale": "Region-specific TV preference."
    })},

    # Speculative umbrella (your dataset has only the hybrid)
    {"role": "user", "content": "I want something imaginative with futuristic tech or magic."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Sci-Fi & Fantasy",
        "confidence": 0.7,
        "rationale": "Speculative elements fit the hybrid label available in the dataset."
    })},

    # Music/Musicals
    {"role": "user", "content": "Concert films or song-driven stories would be great."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Music & Musicals",
        "confidence": 0.8,
        "rationale": "Music-centric content preference."
    })},

    # Reality TV
    {"role": "user", "content": "Unscripted competitions and makeovers, please."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Reality TV",
        "confidence": 0.87,
        "rationale": "Unscripted, competition/makeover cues indicate reality programming."
    })},

    # Ambiguous → broad label with lower confidence
    {"role": "user", "content": "Surprise me with something intense and imaginative."},
    {"role": "assistant", "content": json.dumps({
        "genre": "Sci-Fi & Fantasy",
        "confidence": 0.6,
        "rationale": "Imaginative intensity suggests speculative elements; broad hybrid label chosen."
    })},
]

GENRE_SYSTEM_PROMPT = f"""
    You are a movie/TV genre classifier.
    Return JSON ONLY matching this schema:
    {{"genre": "<one of {ALLOWED_GENRES}>", "confidence": <0.0-1.0>, "rationale": "<=240 chars, no spoilers>"}}

    Rules:
    - Select EXACTLY ONE primary label from the allowed list (dataset-faithful).
    - Choose the label that best reflects the story driver and tone.
    - If the request implies TV vs Movie, prefer the matching TV*/Movies label.
    - Tie-breaks:
    • Non-fiction presentation → "Documentaries" or "Docuseries".
    • Horror vs Thriller: sustained fear/supernatural → "Horror Movies"; crime/tension-driven → "Thrillers".
    • Action/Crime tension with set-pieces (car chases, shootouts, stunts) → "TV Action & Adventure" (or closest action label available).
    • Sci-Fi vs Fantasy: science/tech → "Sci-Fi & Fantasy" (if that’s the only sci-fi/fantasy option); mythic/magical → "Sci-Fi & Fantasy" as well (choose it when speculative elements are clear).
    • Family-friendly film preference → "Children & Family Movies".
    • Bingeable global series → "International TV Shows" (or regional variants like "British TV Shows", "Korean TV Shows" if specified).
    - If unclear, choose the broadest applicable dataset label and lower confidence.
    - Confidence is your probability that the chosen label matches the user’s intent.
    - Rationale: short, user-friendly, cite surface cues (tone, setting, plot driver).
    - No extra keys, no commentary, output must be valid JSON.
    """.strip()

def build_genre_messages(request_text: str):
    return (
        [{"role": "system", "content": GENRE_SYSTEM_PROMPT}]
        + genre_few_shots
        + [{"role": "user", "content": request_text}]
    )

def extract_genre_from_request(request: dict) -> ExtractGenre:
    message = build_genre_messages(str(request))
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
        messages=message,
            response_format=ExtractGenre,
        )
    genre_result = genre_extraction.choices[0].message.parsed
    logger.info(f"Confirmation message generated successfully: {genre_result}")
    return genre_result