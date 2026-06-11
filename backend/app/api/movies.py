from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.movie import Movie, UserWatchlist
from app.schemas.movie import MovieResponse, WatchlistCreate, WatchlistResponse, WatchlistUpdate

router = APIRouter(prefix="/api/movies", tags=["movies"])

@router.get("/search", response_model=List[MovieResponse])
def search_movies(
    q: str = "",
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Direct relational search for movies in PostgreSQL.
    """
    if not q:
        return db.query(Movie).order_by(Movie.popularity.desc()).limit(limit).all()
        
    return db.query(Movie).filter(
        Movie.title.ilike(f"%{q}%")
    ).limit(limit).all()

@router.get("/watchlist", response_model=List[WatchlistResponse])
def get_watchlist(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Fetch the authenticated user's complete watchlist.
    """
    return db.query(UserWatchlist).filter(UserWatchlist.user_id == current_user.id).all()

@router.post("/watchlist", response_model=WatchlistResponse, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(
    watchlist_in: WatchlistCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a movie to the watchlist or update its watch status.
    """
    # Verify the movie exists in local PostgreSQL cache
    movie = db.query(Movie).filter(Movie.id == watchlist_in.movie_id).first()
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Movie not found in database."
        )
        
    # Check if already in watchlist
    item = db.query(UserWatchlist).filter(
        UserWatchlist.user_id == current_user.id,
        UserWatchlist.movie_id == watchlist_in.movie_id
    ).first()
    
    if item:
        item.status = watchlist_in.status
        db.commit()
        db.refresh(item)
        return item
        
    new_item = UserWatchlist(
        user_id=current_user.id,
        movie_id=watchlist_in.movie_id,
        status=watchlist_in.status
    )
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item

@router.delete("/watchlist/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_watchlist(
    movie_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a movie from the user's watchlist.
    """
    item = db.query(UserWatchlist).filter(
        UserWatchlist.user_id == current_user.id,
        UserWatchlist.movie_id == movie_id
    ).first()
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist entry not found."
        )
        
    db.delete(item)
    db.commit()
    return None
