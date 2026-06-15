import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Float, Integer, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    overall_score = Column(Float, nullable=True)
    pass_rate = Column(Float, nullable=True)
    total_cases = Column(Integer, default=0)
    passed_cases = Column(Integer, default=0)
    failed_cases = Column(Integer, default=0)
    
    # Category Average Scores
    recommendation_score = Column(Float, nullable=True)
    personality_score = Column(Float, nullable=True)
    memory_score = Column(Float, nullable=True)
    retrieval_score = Column(Float, nullable=True)
    refusal_score = Column(Float, nullable=True)

    # Category Pass Rates
    recommendation_pass_rate = Column(Float, nullable=True)
    personality_pass_rate = Column(Float, nullable=True)
    memory_pass_rate = Column(Float, nullable=True)
    retrieval_pass_rate = Column(Float, nullable=True)
    refusal_pass_rate = Column(Float, nullable=True)

    status = Column(String(20), default="running")  # "running", "completed", "failed"

    # Relationships
    case_results = relationship("EvaluationCaseResult", back_populates="run", cascade="all, delete-orphan")


class EvaluationCaseResult(Base):
    __tablename__ = "evaluation_case_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False)
    case_id = Column(String(50), nullable=False)
    category = Column(String(50), nullable=False)  # "recommendation", "personality", "memory", "retrieval", "refusal"
    difficulty = Column(String(20), nullable=True)
    query = Column(Text, nullable=False)
    expected = Column(JSONB, nullable=True)
    actual_response = Column(Text, nullable=True)
    passed = Column(Boolean, nullable=False)
    score = Column(Float, nullable=False)
    
    # Sub-scores structure (e.g. relevance, personalization, humor, etc.)
    sub_scores = Column(JSONB, nullable=True)
    strengths = Column(JSONB, nullable=True)
    weaknesses = Column(JSONB, nullable=True)
    reasoning = Column(Text, nullable=True)

    # Relationships
    run = relationship("EvaluationRun", back_populates="case_results")
