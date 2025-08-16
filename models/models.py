from pydantic import BaseModel, Field
from typing import Optional 


class ExtractGenre(BaseModel):
    genre: Optional[list[str]] = Field(
        description="List of genres to filter movies by",
        default_factory=list,
        example=["Action", "Comedy", "Drama"]
    )

class ExtractViewingType(BaseModel):
    viewing_type: Optional[list[str]] = Field(
        description="List of the viewing types to filter movies by",
        default_factory=list,
        example=["Movie", "TV Show", "Miniseries"]
    )


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