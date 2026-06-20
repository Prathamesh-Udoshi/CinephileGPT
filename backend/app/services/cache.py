import logging
import json
import hashlib
from typing import Optional, List, Any
import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger("cache_service")

class CachedChatMessage:
    """
    Lightweight model representing a chat message retrieved from the cache.
    Ensures dot-access compatibility (msg.role, msg.content) with database models.
    """
    def __init__(self, role: str, content: str, timestamp: str = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp

class BaseCacheService:
    async def get_recommendation(self, key: str) -> Optional[dict]:
        raise NotImplementedError()

    async def set_recommendation(self, key: str, value: dict, ttl: int = None) -> None:
        raise NotImplementedError()

    async def get_session_messages(self, session_id: str) -> Optional[List[CachedChatMessage]]:
        raise NotImplementedError()

    async def set_session_messages(self, session_id: str, messages: List[Any], ttl: int = None) -> None:
        raise NotImplementedError()

    async def invalidate_session(self, session_id: str) -> None:
        raise NotImplementedError()

    def generate_recommendation_key(self, profile: dict) -> str:
        raise NotImplementedError()


class NoOpCacheService(BaseCacheService):
    """
    Fallback implementation that behaves as a transparent pass-through when Redis is disabled or unavailable.
    """
    async def get_recommendation(self, key: str) -> Optional[dict]:
        return None

    async def set_recommendation(self, key: str, value: dict, ttl: int = None) -> None:
        pass

    async def get_session_messages(self, session_id: str) -> Optional[List[CachedChatMessage]]:
        return None

    async def set_session_messages(self, session_id: str, messages: List[Any], ttl: int = None) -> None:
        pass

    async def invalidate_session(self, session_id: str) -> None:
        pass

    def generate_recommendation_key(self, profile: dict) -> str:
        return ""


class RedisCacheService(BaseCacheService):
    """
    Production-grade Redis implementation for caching recommendations and user sessions.
    Handles connection loss gracefully by falling back to pass-through mode.
    """
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.enabled = settings.ENABLE_REDIS_CACHE

    async def connect(self):
        if not self.enabled:
            logger.info("Redis cache is disabled in configurations.")
            return
        try:
            logger.info(f"Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT} (DB {settings.REDIS_DB})...")
            self.redis = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=2.0, # 2 seconds connect timeout to avoid blocking startup
                protocol=2 # Force RESP2 for compatibility with older Redis 5.x versions on Windows
            )
            await self.redis.ping()
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Gracefully falling back to Cache-disabled mode.")
            self.redis = None

    async def disconnect(self):
        if self.redis:
            try:
                await self.redis.aclose()
                logger.info("Closed Redis connection.")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            self.redis = None

    def is_active(self) -> bool:
        return self.enabled and self.redis is not None

    def generate_recommendation_key(self, profile: dict) -> str:
        """
        Deduplicates, sorts, and serializes user preference profiles.
        Generates a stable SHA-256 hash to use as a cache key.
        """
        normalized = {}
        for key, value in profile.items():
            if value is None:
                continue
            
            # Clean and normalize lists
            if isinstance(value, list):
                clean_list = []
                for item in value:
                    if isinstance(item, str):
                        clean_list.append(item.strip().lower())
                    elif isinstance(item, int):
                        clean_list.append(item)
                
                # Deduplicate and sort list items
                sorted_list = sorted(list(set(clean_list)), key=lambda x: str(x))
                if sorted_list:
                    normalized[key] = sorted_list
            
            # Clean strings
            elif isinstance(value, str):
                cleaned_str = value.strip().lower()
                if cleaned_str:
                    normalized[key] = cleaned_str
            
            # Keep numbers/booleans as-is
            elif isinstance(value, (int, float, bool)):
                normalized[key] = value

        # Serialize deterministically with sorted keys
        serialized = json.dumps(normalized, sort_keys=True)
        hash_digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        return f"cinephile:recommendation:{hash_digest}"

    async def get_recommendation(self, key: str) -> Optional[dict]:
        if not self.is_active() or not key:
            return None
        try:
            val = await self.redis.get(key)
            if val:
                logger.info(f"Cache Hit for recommendation key: {key}")
                return json.loads(val)
        except Exception as e:
            logger.error(f"Redis get_recommendation error: {e}")
        return None

    async def set_recommendation(self, key: str, value: dict, ttl: int = None) -> None:
        if not self.is_active() or not key:
            return
        if ttl is None:
            ttl = settings.REDIS_TTL_RECOMMENDATIONS
        try:
            logger.info(f"Caching recommendation response under key: {key} (TTL: {ttl}s)")
            await self.redis.set(key, json.dumps(value), ex=ttl)
        except Exception as e:
            logger.error(f"Redis set_recommendation error: {e}")

    async def get_session_messages(self, session_id: str) -> Optional[List[CachedChatMessage]]:
        if not self.is_active() or not session_id:
            return None
        key = f"cinephile:session:{session_id}:messages"
        try:
            val = await self.redis.get(key)
            if val:
                logger.info(f"Cache Hit for session history: {session_id}")
                raw_messages = json.loads(val)
                return [
                    CachedChatMessage(
                        role=msg["role"],
                        content=msg["content"],
                        timestamp=msg.get("timestamp")
                    )
                    for msg in raw_messages
                ]
        except Exception as e:
            logger.error(f"Redis get_session_messages error: {e}")
        return None

    async def set_session_messages(self, session_id: str, messages: List[Any], ttl: int = None) -> None:
        if not self.is_active() or not session_id:
            return
        if ttl is None:
            ttl = settings.REDIS_TTL_SESSIONS
        key = f"cinephile:session:{session_id}:messages"
        try:
            logger.info(f"Caching session history for session: {session_id} (TTL: {ttl}s)")
            # Serialize SQLAlchemy message models or dictionaries into clean JSON
            serialized = []
            for msg in messages:
                # Handle both SQLAlchemy objects and raw dictionaries
                role = getattr(msg, "role", None) or msg.get("role")
                content = getattr(msg, "content", None) or msg.get("content")
                timestamp = getattr(msg, "timestamp", None)
                timestamp_str = timestamp.isoformat() if timestamp and hasattr(timestamp, "isoformat") else str(timestamp) if timestamp else None
                
                serialized.append({
                    "role": role,
                    "content": content,
                    "timestamp": timestamp_str
                })
            await self.redis.set(key, json.dumps(serialized), ex=ttl)
        except Exception as e:
            logger.error(f"Redis set_session_messages error: {e}")

    async def invalidate_session(self, session_id: str) -> None:
        if not self.is_active() or not session_id:
            return
        key = f"cinephile:session:{session_id}:messages"
        try:
            logger.info(f"Invalidating session history cache for session: {session_id}")
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis invalidate_session error: {e}")

# Global cache service instances
cache_service = RedisCacheService()
noop_cache_service = NoOpCacheService()

def get_cache_service() -> BaseCacheService:
    """
    Dependency injection function returning the active cache service or the No-Op fallback.
    """
    if not settings.ENABLE_REDIS_CACHE or not cache_service.is_active():
        return noop_cache_service
    return cache_service
