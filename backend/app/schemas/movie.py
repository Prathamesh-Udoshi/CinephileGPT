from datetime import date, datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field

class MovieBase(BaseModel):
    id: int
    title: str
    release_date: Optional[date] = None
    director: Optional[str] = None
    cast_members: List[str] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    overview: Optional[str] = None
    runtime: Optional[int] = None
    vote_average: Optional[float] = None
    popularity: Optional[float] = None
    poster_path: Optional[str] = None

class MovieResponse(MovieBase):
    class Config:
        from_attributes = True

class WatchlistCreate(BaseModel):
    movie_id: int
    status: str = Field(default="watchlist", description="'watchlist', 'watched', 'liked', or 'disliked'")

class WatchlistUpdate(BaseModel):
    status: str = Field(..., description="'watchlist', 'watched', 'liked', or 'disliked'")

class WatchlistResponse(BaseModel):
    id: UUID
    user_id: UUID
    movie_id: int
    status: str
    updated_at: datetime
    movie: MovieResponse

    class Config:
        from_attributes = True
