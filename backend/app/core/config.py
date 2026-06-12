import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Load settings from a .env file if it exists
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/cinephile_db",
        validation_alias="DATABASE_URL"
    )
    
    GEMINI_API_KEY: str = Field(
        default="",
        validation_alias="GEMINI_API_KEY"
    )
    
    GEMINI_MODEL_NAME: str = Field(
        default="gemini-1.5-flash",
        validation_alias="GEMINI_MODEL_NAME"
    )
    
    JWT_SECRET: str = Field(
        default="your_super_secret_jwt_key_change_me_in_production",
        validation_alias="JWT_SECRET"
    )
    
    JWT_ALGORITHM: str = Field(
        default="HS256",
        validation_alias="JWT_ALGORITHM"
    )
    
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=1440,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    QDRANT_PATH: str = Field(
        default="./qdrant_db",
        validation_alias="QDRANT_PATH"
    )
    
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL_NAME"
    )
    
    OMDB_API_KEY: str = Field(
        default="",
        validation_alias="OMDB_API_KEY"
    )

settings = Settings()
