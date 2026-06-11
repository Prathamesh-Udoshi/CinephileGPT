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
