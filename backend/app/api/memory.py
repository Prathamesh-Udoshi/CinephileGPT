from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.memory import UserProfile
from app.schemas.memory import UserProfileResponse, UserProfileUpdate

router = APIRouter(prefix="/api/memory", tags=["memory"])

@router.get("/profile", response_model=UserProfileResponse)
def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found."
        )
    return profile

@router.put("/profile", response_model=UserProfileResponse)
def update_profile(
    profile_in: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found."
        )
        
    # Update fields if provided
    if profile_in.favorite_genres is not None:
        profile.favorite_genres = profile_in.favorite_genres
    if profile_in.favorite_directors is not None:
        profile.favorite_directors = profile_in.favorite_directors
    if profile_in.favorite_actors is not None:
        profile.favorite_actors = profile_in.favorite_actors
    if profile_in.disliked_genres is not None:
        profile.disliked_genres = profile_in.disliked_genres
    if profile_in.general_notes is not None:
        profile.general_notes = profile_in.general_notes
        
    db.commit()
    db.refresh(profile)
    return profile
