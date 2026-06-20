from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.services.analytics import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("")
def get_analytics_report(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns high-level system caching efficiency, latencies, savings, and preference insights.
    Accessible to authenticated users.
    """
    return AnalyticsService.get_metrics(db)
