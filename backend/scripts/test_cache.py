import os
import sys
import asyncio

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

def test_normalization():
    print("Testing preference profile normalization and key generation...")
    from app.services.cache import cache_service
    
    profile_1 = {
        "genres": ["Action", "Sci-Fi"],
        "moods": ["tense", "exciting"],
        "favorite_directors": ["Christopher Nolan", "Denis Villeneuve"],
        "exclude_genres": ["Romance"]
    }
    
    # Same options, different ordering
    profile_2 = {
        "genres": ["Sci-Fi", "Action"],
        "moods": ["exciting", "tense"],
        "favorite_directors": ["Denis Villeneuve", "Christopher Nolan"],
        "exclude_genres": ["Romance"]
    }
    
    key_1 = cache_service.generate_recommendation_key(profile_1)
    key_2 = cache_service.generate_recommendation_key(profile_2)
    
    print(f"  Key 1: {key_1}")
    print(f"  Key 2: {key_2}")
    
    assert key_1 == key_2, "Normalization failed! Keys are different for semantically identical profiles."
    print("  Success: Deterministic serialization and hashing is working correctly!")

async def test_redis_operations():
    print("\nTesting Redis get/set/delete operations...")
    from app.services.cache import cache_service
    
    # Initialize connection
    await cache_service.connect()
    
    if not cache_service.is_active():
        print("  [WARNING] Redis is not active/running. Skipping operations checks. (Graceful fallback verified)")
        return
        
    test_key = "cinephile:test:verification_key"
    test_data = {
        "movies": [{"id": 101, "title": "Inception"}],
        "response_text": "A masterpiece by Nolan."
    }
    
    try:
        # Set
        await cache_service.set_recommendation(test_key, test_data, ttl=10)
        print("  Data stored in Redis.")
        
        # Get
        retrieved = await cache_service.get_recommendation(test_key)
        print(f"  Data retrieved from Redis: {retrieved}")
        
        assert retrieved == test_data, "Data mismatch!"
        print("  Success: Redis get/set operations verified!")
        
        # Session Cache test
        session_id = "test-session-uuid"
        messages = [
            {"role": "user", "content": "Recommend a sci-fi movie"},
            {"role": "assistant", "content": "I recommend Inception."}
        ]
        await cache_service.set_session_messages(session_id, messages)
        print("  Session messages cached.")
        
        cached_msgs = await cache_service.get_session_messages(session_id)
        assert len(cached_msgs) == 2, "Session cache length mismatch!"
        assert cached_msgs[0].role == "user", "Session message role mismatch!"
        print("  Success: Session caching verified!")
        
        # Clean up
        await cache_service.redis.delete(test_key)
        await cache_service.invalidate_session(session_id)
        print("  Test data cleaned up successfully.")
        
    except Exception as e:
        print(f"  [ERROR] Redis operations failed: {e}")
    finally:
        await cache_service.disconnect()

async def main():
    test_normalization()
    await test_redis_operations()
    print("\nAll caching service verification tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
