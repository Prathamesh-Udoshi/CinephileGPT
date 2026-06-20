from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from qdrant_client import QdrantClient
from app.core.config import settings

# PostgreSQL Setup
engine = create_engine(
    settings.DATABASE_URL,
    # pool_pre_ping helps recover from database restarts or dropped connections
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Local file-based Qdrant Client Setup (Zero Docker dependency)
_qdrant_client = None

def get_db():
    """
    SQLAlchemy database session dependency.
    Closes the session automatically after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_qdrant() -> QdrantClient:
    """
    Lazy-load and return the Qdrant client instance to avoid lock conflicts during startup/reloading.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
    return _qdrant_client

def init_db_schema():
    """
    Ensures all tables (including application, logging, and evaluation models) are created.
    Runs direct SQL ALTER queries to migrate existing tables with new fields.
    """
    # Import all models locally to register them in Base.metadata
    from app.models.user import User
    from app.models.movie import Movie, UserWatchlist
    from app.models.memory import UserProfile, ChatSession, ChatMessage, RecommendationLog, UserFeedback
    try:
        from evaluation.models import EvaluationRun, EvaluationCaseResult
    except ImportError:
        pass

    # Create tables if they do not exist
    Base.metadata.create_all(bind=engine)

    # Database migration for existing recommendation_logs table: add columns if not exists
    from sqlalchemy import text
    with engine.begin() as conn:
        columns_to_add = [
            ("original_query", "TEXT"),
            ("cache_hit", "BOOLEAN DEFAULT FALSE"),
            ("retrieval_latency_ms", "DOUBLE PRECISION"),
            ("llm_provider", "VARCHAR(50)"),
            ("llm_latency_ms", "DOUBLE PRECISION"),
            ("total_response_time_ms", "DOUBLE PRECISION"),
            ("error_message", "TEXT")
        ]
        for col_name, col_type in columns_to_add:
            try:
                # Add columns if not exists
                conn.execute(text(f"ALTER TABLE recommendation_logs ADD COLUMN IF NOT EXISTS {col_name} {col_type}"))
            except Exception as e:
                import logging
                logging.getLogger("database").warning(f"Could not check/add column {col_name}: {e}")

