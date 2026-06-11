import uuid
from sqlalchemy import Column, String, Integer, Date, Text, Numeric, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True) # TMDB Movie ID
    title = Column(String(255), nullable=False, index=True)
    release_date = Column(Date)
    director = Column(String(255), index=True)
    cast_members = Column(ARRAY(String))
    genres = Column(ARRAY(String))
    overview = Column(Text)
    runtime = Column(Integer)
    vote_average = Column(Numeric(3, 1))
    popularity = Column(Numeric(8, 2))
    poster_path = Column(String(255))

    # Relationships
    watchlists = relationship("UserWatchlist", back_populates="movie", cascade="all, delete-orphan")


class UserWatchlist(Base):
    __tablename__ = "user_watchlists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), default="watchlist")  # 'watchlist', 'watched', 'liked', 'disliked'
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="watchlist")
    movie = relationship("Movie", back_populates="watchlists")

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_watchlist"),
    )
