from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel

class ChatMessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True

class ChatSessionResponse(BaseModel):
    id: UUID
    created_at: datetime
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

class ChatStreamRequest(BaseModel):
    session_id: Optional[UUID] = None
    message: str

class FeedbackRequest(BaseModel):
    recommendation_log_id: UUID
    rating: int  # 1 for upvote, -1 for downvote
    feedback_text: Optional[str] = None

