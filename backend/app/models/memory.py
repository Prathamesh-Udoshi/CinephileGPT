import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Boolean, Float
from sqlalchemy.dialects.postgresql import ARRAY, UUID, JSONB

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    favorite_genres = Column(ARRAY(String), default=list)
    favorite_directors = Column(ARRAY(String), default=list)
    favorite_actors = Column(ARRAY(String), default=list)
    disliked_genres = Column(ARRAY(String), default=list)
    general_notes = Column(Text, default="")
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="profile")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # State representation of known information (preferences, notes, query state)
    known_information = Column(JSONB, nullable=True)
    
    # Process attributes
    information_complete = Column(Boolean, default=False)
    follow_up_questions = Column(ARRAY(String), nullable=True)
    
    # Search & recommendations
    generated_query = Column(Text, nullable=True)
    retrieved_movie_ids = Column(ARRAY(Integer), nullable=True)
    recommended_movie_ids = Column(ARRAY(Integer), nullable=True)
    explanations = Column(JSONB, nullable=True)

    # Logging and metrics tracking
    original_query = Column(Text, nullable=True)
    cache_hit = Column(Boolean, default=False)
    retrieval_latency_ms = Column(Float, nullable=True)
    llm_provider = Column(String(50), nullable=True)
    llm_latency_ms = Column(Float, nullable=True)
    total_response_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)


class UserFeedback(Base):
    __tablename__ = "user_feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_log_id = Column(UUID(as_uuid=True), ForeignKey("recommendation_logs.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1 = upvote, -1 = downvote
    feedback_text = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User")
    recommendation_log = relationship("RecommendationLog")

