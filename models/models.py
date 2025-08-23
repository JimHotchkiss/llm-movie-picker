from pydantic import BaseModel, Field, field_validator
from typing import ClassVar
from typing import Optional 


ALLOWED_GENRES = [
    "Documentaries",
    "Docuseries",
    "Reality TV",

    "International TV Shows",
    "British TV Shows",
    "Korean TV Shows",
    "Kids' TV",
    "Teen TV Shows",
    "Talk Shows",
    "Sports Series",
    "Science & Nature TV",

    "TV Dramas",
    "TV Mysteries",
    "TV Comedies",
    "TV Action & Adventure",

    "Children & Family Movies",
    "Independent Movies",
    "Classic Movies",
    "Anime Features",
    "LGBTQ Movies",

    "Horror Movies",
    "Thrillers",
    "Comedy",           # (present in your tokens as Stand-Up/Comedy)
    "Music & Musicals", # ("Music", "Musicals" tokens)
    "Cult Movies",

    "Spanish-Language", # (can be applied platform-wide; keep as-is if your dataset has it)
    "Faith & Spirituality",
    "Sci-Fi & Fantasy",
    # If your dataset *also* has singular splits, you can keep these too:
    # "Horror", "Sci-Fi", "Fantasy"
]




class ExtractGenre(BaseModel):
    genre: str = Field(description="Select a single genre from the dataset labels")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    rationale: str = Field(default="", max_length=240, description="Short reasion for choice")

    # store valid genres for reuse
    allowed_genres: ClassVar[list[str]] = ALLOWED_GENRES

    @field_validator("genre")
    def validate_genre(cls, v: str) -> str:
        if v not in cls.allowed_genres:
            raise ValueError(f"Genre '{v}' not in allowed list: {cls.allowed_genres}")
        return v
  

class ExtractViewingType(BaseModel):
    viewing_type: Optional[list[str]] = Field(
        description="List of the viewing types to filter movies by",
        default_factory=list,
        example=["Movie", "TV Show", "Miniseries"]
    )

class ExtractAudienceCategory(BaseModel):
    category: str = Field(description="One of CHILDREN, TEEN, ADULT, or '' when unknown", default="")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0–1.0")
    rationale: str = Field(default="", description="Short reason for choice")


class ExtractDescription(BaseModel):
    description: Optional[str] = Field(
        description="Extract the description of the movie",
        example="A thrilling adventure movie"
    )

# VAD model for valence, arousal, and dominance
# This model can be used to represent the emotional state of a movie or user preference
# Valence: Pleasantness (0–1)
# Arousal: Activation (0–1)
# Dominance: Control (0–1)
# Each field is constrained to be between 0.0 and 1.0
# With 'ge' and 'le' meaning greater than or less than
class VAD(BaseModel):
    valence: float = Field(..., ge=0.0, le=1.0, description="Pleasantness (0–1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Activation (0–1)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Control (0–1)")

class ExtractVAD(BaseModel):
    vad: VAD 
    model_config = {"extra": "forbid"}  # Disallow extra fields

class MovieVAD(BaseModel):
    valence: float = Field(..., ge=0.0, le=1.0, description="Pleasantness (0–1)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Activation (0–1)")
    dominance: float = Field(..., ge=0.0, le=1.0, description="Control (0–1)")

class ExtractMovieVAD(BaseModel):
    movie_vad: MovieVAD 
    model_config = {"extra": "forbid"}  # Disallow extra fields