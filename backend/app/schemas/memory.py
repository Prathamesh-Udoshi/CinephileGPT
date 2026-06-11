from typing import List, Optional
from pydantic import BaseModel, Field

class UserProfileBase(BaseModel):
    favorite_genres: List[str] = Field(default_factory=list)
    favorite_directors: List[str] = Field(default_factory=list)
    favorite_actors: List[str] = Field(default_factory=list)
    disliked_genres: List[str] = Field(default_factory=list)
    general_notes: str = ""

class UserProfileResponse(UserProfileBase):
    class Config:
        from_attributes = True

class UserProfileUpdate(BaseModel):
    favorite_genres: Optional[List[str]] = None
    favorite_directors: Optional[List[str]] = None
    favorite_actors: Optional[List[str]] = None
    disliked_genres: Optional[List[str]] = None
    general_notes: Optional[str] = None
