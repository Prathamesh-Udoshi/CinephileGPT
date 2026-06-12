import os
import sys
import csv
import urllib.request
from datetime import datetime, date
from sqlalchemy.orm import Session

# Add the backend directory to Python path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.database import Base, engine, SessionLocal, get_qdrant
from app.models.movie import Movie
from app.services.retrieval import upsert_movies_to_qdrant

DATASET_URL = "https://raw.githubusercontent.com/maazh/IMDB-Movie-Dataset-Analysis/master/tmdb-movies.csv"

def parse_csv_date(date_str: str) -> date:
    """
    Parse date string in various possible formats from the CSV.
    Handles 2-digit years safely assuming <= 26 is 2000s and > 26 is 1900s.
    """
    if not date_str or date_str == "N/A":
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            dt = datetime.strptime(date_str, fmt).date()
            if dt.year < 100:
                year = dt.year
                if year <= 26:
                    year += 2000
                else:
                    year += 1900
                dt = date(year, dt.month, dt.day)
            return dt
        except ValueError:
            continue
    return None

def download_and_parse_dataset(limit: int = 1500) -> list:
    """
    Download TMDB CSV from raw repository and parse top N most popular movies.
    """
    print(f"Downloading TMDB movie dataset from:\n  {DATASET_URL}")
    try:
        response = urllib.request.urlopen(DATASET_URL)
        csv_data = response.read().decode('utf-8').splitlines()
        print(f"Successfully downloaded. Parsing {len(csv_data)} rows...")
    except Exception as e:
        print(f"[ERROR] Failed to download movie dataset: {e}")
        return []

    reader = csv.DictReader(csv_data)
    movies_list = []

    for row in reader:
        # Check required fields
        title = row.get("original_title") or row.get("title")
        overview = row.get("overview")
        m_id_str = row.get("id")
        
        if not title or not overview or not m_id_str:
            continue

        try:
            m_id = int(m_id_str)
            popularity = float(row.get("popularity", 0))
            vote_average = float(row.get("vote_average", 0))
            runtime = int(row.get("runtime")) if row.get("runtime") else None
        except ValueError:
            continue

        # Parse comma/pipe-separated fields
        genres_raw = row.get("genres", "")
        genres = [g.strip() for g in genres_raw.split("|") if g.strip()] if genres_raw else []

        cast_raw = row.get("cast", "")
        cast = [c.strip() for c in cast_raw.split("|") if c.strip()][:5] if cast_raw else []

        director_raw = row.get("director", "")
        director = director_raw.replace("|", ", ").strip()
        if len(director) > 255:
            director = director[:252] + "..."

        release_date = parse_csv_date(row.get("release_date"))

        movies_list.append({
            "id": m_id,
            "title": title,
            "release_date": release_date,
            "director": director or "Unknown",
            "cast_members": cast,
            "genres": genres,
            "overview": overview,
            "runtime": runtime,
            "vote_average": vote_average,
            "popularity": popularity
        })

    # Sort by popularity descending and keep top N
    movies_list.sort(key=lambda x: x["popularity"], reverse=True)
    top_movies = movies_list[:limit]
    print(f"Parsed and sorted. Filtered to top {len(top_movies)} most popular movies.")
    return top_movies

def run_all_time_seeding(limit: int = 1500) -> int:
    """
    Seeding runner for all-time popular movies database.
    """
    print("\n=== Starting All-Time English Movies Seeding Pipeline ===")
    
    # Download and parse
    movies_to_seed = download_and_parse_dataset(limit=limit)
    if not movies_to_seed:
        print("[Error] No movie records to seed. Aborting.")
        return 0

    # Connect to PostgreSQL
    try:
        Base.metadata.create_all(bind=engine)
        db: Session = SessionLocal()
    except Exception as e:
        print(f"[Error] Failed to connect to PostgreSQL database: {e}")
        return 0

    print("Syncing movie records to PostgreSQL database...")
    synced_movies = []
    
    for m_data in movies_to_seed:
        try:
            existing = db.query(Movie).filter(Movie.id == m_data["id"]).first()
            if existing:
                # Update attributes
                for k, v in m_data.items():
                    setattr(existing, k, v)
                db_movie = existing
            else:
                db_movie = Movie(**m_data)
                db.add(db_movie)
            
            db.commit()
            synced_movies.append(db_movie)
        except Exception as e:
            db.rollback()
            print(f"  [Warning] Failed to save movie '{m_data['title']}' (ID: {m_data['id']}): {e}")

    print(f"PostgreSQL database synchronization complete. Synced {len(synced_movies)} movies.")

    # Index into Qdrant in batches of 150
    if synced_movies:
        print(f"\nIndexing {len(synced_movies)} movies into Qdrant local vector database...")
        try:
            qdrant_client = get_qdrant()
            
            # Batch size of 150 is optimal for memory and CPU usage during local embedding extraction
            batch_size = 150
            for i in range(0, len(synced_movies), batch_size):
                batch = synced_movies[i:i + batch_size]
                print(f"  Indexing batch {i // batch_size + 1}/{-(-len(synced_movies) // batch_size)} ({len(batch)} movies)...")
                upsert_movies_to_qdrant(qdrant_client, batch)
                
            print("Qdrant local vector indexing complete!")
        except Exception as e:
            print(f"[Error] Failed to index vectors in Qdrant: {e}")
    
    db.close()
    print(f"=== Seeding Pipeline Finished. Database holds {len(synced_movies)} all-time popular movies! ===\n")
    return len(synced_movies)

if __name__ == "__main__":
    limit = 1500
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"[Warning] Invalid limit argument '{sys.argv[1]}', using default: {limit}")
    run_all_time_seeding(limit=limit)
