import os
import sys
import json
import httpx
from datetime import datetime, date
import google.generativeai as genai
from sqlalchemy.orm import Session

# Add the backend directory to Python path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.database import Base, engine, SessionLocal, get_qdrant
from app.models.movie import Movie
from app.services.retrieval import upsert_movies_to_qdrant

def generate_movie_list() -> list:
    """
    Use Gemini API to generate a list of 25 popular or acclaimed movies released between 2024 and 2026.
    Falls back to Groq API if Gemini fails.
    """
    prompt = """Generate a JSON list of 25 highly popular or acclaimed movies released between 2024 and 2026.
For each movie, output a JSON object containing the exact title of the movie and its release year.
Make sure to include a diverse set of genres and highly anticipated or released titles (e.g. Dune: Part Two, Furiosa, Deadpool & Wolverine, Inside Out 2, Gladiator II, Oppenheimer, etc.).
Format example:
[
  {"title": "Dune: Part Two", "year": 2024},
  {"title": "Furiosa: A Mad Max Saga", "year": 2024},
  {"title": "Inside Out 2", "year": 2024}
]
Output ONLY a valid JSON list. Do not include any markdown styling, backticks, or explanation."""

    # 1. Try Gemini
    if settings.GEMINI_API_KEY:
        print("Querying Gemini API to generate a list of recent movie titles (2024-2026)...")
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(
                settings.GEMINI_MODEL_NAME,
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
            )
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            try:
                text = response.text
            except (ValueError, IndexError) as error:
                raise ValueError(f"Empty or blocked response ({error})")
            movie_list = json.loads(text.strip())
            print(f"Generated list of {len(movie_list)} movies from Gemini.")
            return movie_list
        except Exception as e:
            print(f"[Warning] Failed to generate movie list from Gemini: {e}. Trying Groq fallback...")

    # 2. Try Groq fallback
    if settings.GROQ_API_KEY:
        print("Querying Groq API fallback to generate a list of recent movie titles (2024-2026)...")
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": settings.GROQ_MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.5
            }
            r = httpx.post(url, headers=headers, json=payload, timeout=20.0)
            if r.status_code == 200:
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                movie_list = json.loads(content.strip())
                # If the LLM returns a dictionary instead of a list, try to extract the list
                if isinstance(movie_list, dict):
                    for val in movie_list.values():
                        if isinstance(val, list):
                            movie_list = val
                            break
                if isinstance(movie_list, list):
                    print(f"Generated list of {len(movie_list)} movies from Groq fallback.")
                    return movie_list
                else:
                    print(f"[Error] Groq response did not parse as list: {movie_list}")
            else:
                print(f"[Error] Groq API returned {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[ERROR] Failed to generate movie list from Groq fallback: {e}")

    print("[Error] Both Gemini and Groq API calls failed or are unconfigured.")
    return []

def fetch_movie_from_omdb(title: str, year: int) -> dict:
    """
    Fetch movie metadata from OMDb API by title and year.
    """
    if not settings.OMDB_API_KEY:
        print("[Error] OMDB_API_KEY is not set.")
        return None

    url = "http://www.omdbapi.com/"
    params = {
        "t": title,
        "y": year,
        "apikey": settings.OMDB_API_KEY,
        "plot": "full"
    }
    
    try:
        print(f"Fetching from OMDb: '{title}' ({year})...")
        response = httpx.get(url, params=params, timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True":
                return data
            else:
                print(f"  [Warning] OMDb search failed for '{title}' ({year}): {data.get('Error')}")
        else:
            print(f"  [Warning] OMDb API HTTP {response.status_code} for '{title}'")
    except Exception as e:
        print(f"  [Warning] Connection error for '{title}': {e}")
    return None

def parse_omdb_movie(data: dict) -> dict:
    """
    Parse OMDb API response fields into format expected by Movie database model.
    """
    try:
        # Parse IMDb ID to integer
        imdb_id_str = data.get("imdbID", "")
        if imdb_id_str.startswith("tt"):
            movie_id = int(imdb_id_str[2:])
        else:
            print(f"  [Skip] Invalid IMDb ID: {imdb_id_str}")
            return None

        # Parse Release Date
        released_str = data.get("Released", "N/A")
        release_date = None
        if released_str != "N/A":
            try:
                release_date = datetime.strptime(released_str, "%d %b %Y").date()
            except ValueError:
                pass
        
        if not release_date:
            year_str = data.get("Year", "")
            # Handle year ranges like 2020- or 2024-2025
            clean_year = "".join(filter(str.isdigit, year_str))[:4]
            if clean_year:
                release_date = date(int(clean_year), 1, 1)
            else:
                release_date = date.today()

        # Parse Cast Members
        actors_str = data.get("Actors", "Unknown")
        cast_members = [a.strip() for a in actors_str.split(",") if a.strip()] if actors_str != "N/A" else ["Unknown"]

        # Parse Genres
        genres_str = data.get("Genre", "Unknown")
        genres = [g.strip() for g in genres_str.split(",") if g.strip()] if genres_str != "N/A" else ["Unknown"]

        # Parse Runtime
        runtime_str = data.get("Runtime", "N/A")
        runtime = None
        if runtime_str != "N/A":
            clean_runtime = "".join(filter(str.isdigit, runtime_str))
            if clean_runtime:
                runtime = int(clean_runtime)

        # Parse Vote Average (imdbRating)
        rating_str = data.get("imdbRating", "N/A")
        vote_average = None
        if rating_str != "N/A":
            try:
                vote_average = float(rating_str)
            except ValueError:
                pass

        # Parse Popularity based on imdbVotes
        votes_str = data.get("imdbVotes", "0")
        popularity = 0.0
        if votes_str != "N/A" and votes_str != "0":
            try:
                clean_votes = votes_str.replace(",", "")
                popularity = float(clean_votes) / 10000.0  # Normalize to a smaller number
            except ValueError:
                pass

        return {
            "id": movie_id,
            "title": data.get("Title"),
            "release_date": release_date,
            "director": data.get("Director") if data.get("Director") != "N/A" else "Unknown",
            "cast_members": cast_members,
            "genres": genres,
            "overview": data.get("Plot") if data.get("Plot") != "N/A" else "No overview available.",
            "runtime": runtime,
            "vote_average": vote_average,
            "popularity": popularity
        }
    except Exception as e:
        print(f"  [Error] Parsing failed for {data.get('Title')}: {e}")
        return None

def run_sync() -> int:
    """
    Main sync runner. Generates movie list, fetches OMDb metadata, updates databases.
    """
    print("=== Starting OMDb & Gemini Movie Synchronization Pipeline ===")
    
    # Generate movie list from Gemini
    movie_list = generate_movie_list()
    if not movie_list:
        print("[Error] No movie titles generated. Aborting sync.")
        return 0

    # Initialize SQL Database
    try:
        Base.metadata.create_all(bind=engine)
        db: Session = SessionLocal()
    except Exception as e:
        print(f"[Error] Failed to connect to PostgreSQL: {e}")
        return 0

    synced_movies = []
    processed_ids = set()

    for item in movie_list:
        title = item.get("title")
        year = item.get("year")
        if not title:
            continue
            
        omdb_data = fetch_movie_from_omdb(title, year)
        if not omdb_data:
            continue
            
        parsed_movie = parse_omdb_movie(omdb_data)
        if not parsed_movie:
            continue
            
        m_id = parsed_movie["id"]
        if m_id in processed_ids:
            continue
            
        try:
            # Check if movie exists in PostgreSQL
            existing = db.query(Movie).filter(Movie.id == m_id).first()
            if existing:
                print(f"  -> Updating existing movie in PostgreSQL: {parsed_movie['title']}")
                for key, val in parsed_movie.items():
                    setattr(existing, key, val)
                db_movie = existing
            else:
                print(f"  -> Inserting new movie into PostgreSQL: {parsed_movie['title']}")
                db_movie = Movie(**parsed_movie)
                db.add(db_movie)
                
            db.commit()
            synced_movies.append(db_movie)
            processed_ids.add(m_id)
        except Exception as e:
            db.rollback()
            print(f"  [Error] Failed to save movie '{parsed_movie['title']}' to database: {e}")

    # Upsert synced movies to Qdrant Local Client
    if synced_movies:
        print(f"\nIndexing {len(synced_movies)} movies into Qdrant local vector database...")
        try:
            qdrant_client = get_qdrant()
            upsert_movies_to_qdrant(qdrant_client, synced_movies)
            print("Qdrant index update complete!")
        except Exception as e:
            print(f"[Error] Failed to update Qdrant vector database: {e}")
    else:
        print("\nNo movies synced. Skipping vector database indexing.")

    db.close()
    print(f"\n=== Synchronization Pipeline Finished. Synced {len(synced_movies)} movies! ===")
    return len(synced_movies)

if __name__ == "__main__":
    run_sync()
