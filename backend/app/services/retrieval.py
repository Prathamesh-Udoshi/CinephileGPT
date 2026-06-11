from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny, MatchValue
from app.core.database import get_qdrant
from app.services.embeddings import EmbeddingService
from app.models.movie import Movie
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

COLLECTION_NAME = "movies_semantic"

def init_qdrant_collection(client: QdrantClient):
    """
    Initialize the Qdrant semantic movie search collection if it doesn't exist.
    """
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # all-MiniLM-L6-v2 vector dimension
                distance=Distance.COSINE
            )
        )

def upsert_movies_to_qdrant(client: QdrantClient, movies: list[Movie]):
    """
    Generate embeddings for movie synopses and upsert to Qdrant.
    """
    init_qdrant_collection(client)
    
    points = []
    for m in movies:
        # Create a text representation combining title, genres, director, and overview
        genres_str = ", ".join(m.genres) if m.genres else ""
        director_str = m.director if m.director else "Unknown"
        overview_str = m.overview if m.overview else ""
        
        text_to_embed = f"Title: {m.title}\nDirector: {director_str}\nGenres: {genres_str}\nOverview: {overview_str}"
        vector = EmbeddingService.get_embedding(text_to_embed)
        
        points.append(
            PointStruct(
                id=m.id,
                vector=vector,
                payload={
                    "movie_id": m.id,
                    "title": m.title,
                    "genres": m.genres or [],
                    "director": m.director or "",
                    "popularity": float(m.popularity) if m.popularity else 0.0,
                    "vote_average": float(m.vote_average) if m.vote_average else 0.0
                }
            )
        )
        
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )

def search_movies_vector(
    client: QdrantClient, 
    query: str, 
    limit: int = 10, 
    favorite_genres: list[str] = None, 
    disliked_genres: list[str] = None,
    director: str = None
) -> list[dict]:
    """
    Search Qdrant for movies semantically matching the query.
    Applies strict filter rules for genres and directors.
    """
    init_qdrant_collection(client)
    query_vector = EmbeddingService.get_embedding(query)
    
    filter_must = []
    filter_must_not = []
    
    if favorite_genres:
        filter_must.append(
            FieldCondition(
                key="genres",
                match=MatchAny(any=favorite_genres)
            )
        )
        
    if director:
        filter_must.append(
            FieldCondition(
                key="director",
                match=MatchValue(value=director)
            )
        )
        
    if disliked_genres:
        filter_must_not.append(
            FieldCondition(
                key="genres",
                match=MatchAny(any=disliked_genres)
            )
        )
        
    query_filter = None
    if filter_must or filter_must_not:
        query_filter = Filter(
            must=filter_must if filter_must else None,
            must_not=filter_must_not if filter_must_not else None
        )
        
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit
    )
    
    return [
        {
            "movie_id": hit.payload["movie_id"],
            "title": hit.payload["title"],
            "score": hit.score
        }
        for hit in results.points
    ]

def hybrid_retrieval(
    db: Session, 
    client: QdrantClient, 
    query: str, 
    limit: int = 5,
    user_profile: dict = None
) -> list[Movie]:
    """
    Retrieve top movies by blending semantic vector search and DB relational metadata filtering.
    """
    fav_genres = user_profile.get("favorite_genres") if user_profile else None
    disliked_genres = user_profile.get("disliked_genres") if user_profile else None
    
    # Bypass vector search for simple greetings/short inputs
    greetings = ["hello", "hello bro", "hi", "hey", "greetings", "yo", "hola", "howdy", "hello there", "test"]
    clean_query = query.strip().lower().replace("?", "").replace("!", "")
    if clean_query in greetings or len(clean_query) < 3:
        return []
    
    # 1. Fetch matches from Qdrant vector database
    vector_results = search_movies_vector(
        client=client,
        query=query,
        limit=limit * 3, # retrieve candidates
        favorite_genres=fav_genres,
        disliked_genres=disliked_genres
    )
    
    if not vector_results:
        # Fallback to database query by popularity if vector database is empty
        query_db = db.query(Movie)
        if disliked_genres:
            # PostgreSQL Array contains check or exclusion
            for genre in disliked_genres:
                query_db = query_db.filter(~Movie.genres.any(genre))
        return query_db.order_by(Movie.popularity.desc()).limit(limit).all()
        
    movie_ids = [res["movie_id"] for res in vector_results]
    
    # 2. Fetch movie metadata from Postgres DB in the order retrieved
    movies = db.query(Movie).filter(Movie.id.in_(movie_ids)).all()
    
    # Sort the returned list back to match the vector ranking order
    movie_dict = {m.id: m for m in movies}
    ordered_movies = []
    for res in vector_results:
        m_id = res["movie_id"]
        if m_id in movie_dict:
            ordered_movies.append(movie_dict[m_id])
            
    return ordered_movies[:limit]
