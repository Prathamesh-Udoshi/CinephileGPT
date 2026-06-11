import os
import sys
import shutil

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

def run_diagnostics():
    print("--- CinephileGPT Backend Diagnostics ---")
    
    # 1. Config Validation
    print("\n1. Verifying Settings Configuration:")
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"  Gemini Model: {settings.GEMINI_MODEL_NAME}")
    print(f"  Qdrant Path: {settings.QDRANT_PATH}")
    print("  Configuration values loaded successfully.")

    # 2. Package Imports Validation
    print("\n2. Verifying Package Imports:")
    try:
        from fastapi import FastAPI
        from sqlalchemy import create_engine
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        import google.generativeai as genai
        from jose import jwt
        from passlib.context import CryptContext
        from sse_starlette.sse import EventSourceResponse
        
        print("  FastAPI, SQLAlchemy, Qdrant, SentenceTransformers, Gemini SDK, PyJWT, Passlib, SSE-Starlette imported successfully.")
    except ImportError as e:
        print(f"  [ERROR] Import failed: {e}")
        sys.exit(1)

    # 3. Embedding Generation Test
    print("\n3. Testing Local Embedding Service:")
    try:
        from app.services.embeddings import EmbeddingService
        test_text = "Who directed the mind-bending sci-fi movie Inception?"
        print("  Generating test embedding (this will download all-MiniLM-L6-v2 on first run)...")
        embedding = EmbeddingService.get_embedding(test_text)
        
        print(f"  Success! Embedding dimension: {len(embedding)}")
        assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
        print("  Embedding vector assertions passed.")
    except Exception as e:
        print(f"  [ERROR] Local embedding service failed: {e}")
        sys.exit(1)

    # 4. Local Qdrant Storage Test
    print("\n4. Testing Local Qdrant Database (File Storage):")
    test_db_dir = "./test_qdrant_db"
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
        
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # Instantiate a temporary local file-based Qdrant client
        client = QdrantClient(path=test_db_dir)
        collection_name = "test_collection"
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("  Temporary local collection created.")
        
        # Insert a point
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=1,
                    vector=embedding,
                    payload={"title": "Inception", "director": "Christopher Nolan"}
                )
            ]
        )
        print("  Test point upserted successfully.")
        
        # Query the point
        search_results = client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=1
        )
        points = search_results.points
        print(f"  Search retrieval completed. Found matches: {len(points)}")
        assert len(points) == 1, "Expected exactly 1 search match"
        assert points[0].payload["title"] == "Inception", "Expected title to match 'Inception'"
        print(f"  Match validation passed: {points[0].payload['title']} by {points[0].payload['director']} (Score: {points[0].score:.4f})")
        
        client.close()
    except Exception as e:
        print(f"  [ERROR] Local Qdrant operations failed: {e}")
        try:
            client.close()
        except:
            pass
        if os.path.exists(test_db_dir):
            try:
                shutil.rmtree(test_db_dir)
            except:
                pass
        sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists(test_db_dir):
            try:
                shutil.rmtree(test_db_dir)
                print("  Temporary local Qdrant files cleaned up.")
            except Exception as cleanup_err:
                print(f"  [WARNING] Temporary directory cleanup deferred: {cleanup_err}")

    # 5. Intent Classification Dry-run
    print("\n5. Testing Intent Classifier Fallback:")
    try:
        from app.services.intent import IntentClassifierService
        # If API key is empty, it should gracefully fall back to MOVIE_DISCUSSION
        intent = IntentClassifierService.classify_intent("Tell me about Star Wars.")
        print(f"  Current classifier return: {intent}")
    except Exception as e:
        print(f"  [ERROR] Intent classifier failed: {e}")
        sys.exit(1)

    print("\n--- All diagnostic checks passed successfully! ---")

if __name__ == "__main__":
    run_diagnostics()
