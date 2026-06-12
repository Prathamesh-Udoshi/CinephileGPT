import os
import sys
from datetime import date
from sqlalchemy.orm import Session

# Add the backend directory to Python path to run script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.database import Base, engine, SessionLocal, get_qdrant
from app.models.movie import Movie
from app.services.retrieval import upsert_movies_to_qdrant, init_qdrant_collection

# A dataset of iconic movies covering various eras, directors, and genres
SEED_MOVIES = [
    {
        "id": 27205,
        "title": "Inception",
        "release_date": date(2010, 7, 16),
        "director": "Christopher Nolan",
        "cast_members": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page", "Tom Hardy"],
        "genres": ["Sci-Fi", "Action", "Thriller"],
        "overview": "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets, is offered a chance to regain his old life as payment for a task considered to be impossible: inception - the implantation of another person's idea into a target's subconscious.",
        "runtime": 148,
        "vote_average": 8.4,
        "popularity": 125.40,
        "poster_path": "/o0OFlw725ZCXcyu1jV2U151m66t.jpg"
    },
    {
        "id": 680,
        "title": "Pulp Fiction",
        "release_date": date(1994, 9, 10),
        "director": "Quentin Tarantino",
        "cast_members": ["John Travolta", "Samuel L. Jackson", "Uma Thurman", "Bruce Willis"],
        "genres": ["Thriller", "Crime"],
        "overview": "A burger-loving hitman, his philosophical partner, a drug-addled gangster's moll, and a washed-up boxer converge in this sprawling, multi-strand cinematic masterpiece of crime and pop culture.",
        "runtime": 154,
        "vote_average": 8.9,
        "popularity": 95.20,
        "poster_path": "/d5iIlvFJm05Vv5Gj1COi7WkPUas.jpg"
    },
    {
        "id": 238,
        "title": "The Godfather",
        "release_date": date(1972, 3, 14),
        "director": "Francis Ford Coppola",
        "cast_members": ["Marlon Brando", "Al Pacino", "James Caan", "Diane Keaton"],
        "genres": ["Drama", "Crime"],
        "overview": "Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone, barely survives an attempt on his life, his youngest son, Michael, steps in to take care of the killers.",
        "runtime": 175,
        "vote_average": 9.2,
        "popularity": 110.10,
        "poster_path": "/3bhkrj6UGV2yy2g51n9B591gTYY.jpg"
    },
    {
        "id": 157336,
        "title": "Interstellar",
        "release_date": date(2014, 11, 5),
        "director": "Christopher Nolan",
        "cast_members": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain", "Michael Caine"],
        "genres": ["Sci-Fi", "Drama", "Adventure"],
        "overview": "The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage.",
        "runtime": 169,
        "vote_average": 8.4,
        "popularity": 140.60,
        "poster_path": "/gEU2QvE37uO7jbmYZfsfsfsadad.jpg"
    },
    {
        "id": 129,
        "title": "Spirited Away",
        "release_date": date(2001, 7, 20),
        "director": "Hayao Miyazaki",
        "cast_members": ["Rumi Hiiragi", "Miyu Irino", "Mari Natsuki"],
        "genres": ["Animation", "Fantasy", "Family"],
        "overview": "A young girl, Chihiro, wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts. She must work in a bathhouse to find a way to free her parents and return to the human world.",
        "runtime": 125,
        "vote_average": 8.5,
        "popularity": 80.30,
        "poster_path": "/393t4P6HYq7jVOSpS05VdbptIYH.jpg"
    },
    {
        "id": 496243,
        "title": "Parasite",
        "release_date": date(2019, 5, 30),
        "director": "Bong Joon Ho",
        "cast_members": ["Song Kang-ho", "Lee Sun-kyun", "Cho Yeo-jeong", "Choi Woo-shik"],
        "genres": ["Thriller", "Drama", "Comedy"],
        "overview": "All unemployed, Ki-taek's family takes peculiar interest in the wealthy and glamorous Parks for their livelihood until they get entangled in an unexpected incident.",
        "runtime": 132,
        "vote_average": 8.5,
        "popularity": 75.80,
        "poster_path": "/7IiTT05Z212z17kAg4QJ46c2E8y.jpg"
    },
    {
        "id": 62,
        "title": "2001: A Space Odyssey",
        "release_date": date(1968, 4, 2),
        "director": "Stanley Kubrick",
        "cast_members": ["Keir Dullea", "Gary Lockwood", "William Sylvester"],
        "genres": ["Sci-Fi", "Mystery", "Adventure"],
        "overview": "Supercomputer HAL 9000 guides astronauts on a trip to Jupiter, but malfunctions, leading to a tense, philosophical space showdown between man and machine.",
        "runtime": 149,
        "vote_average": 8.1,
        "popularity": 45.30,
        "poster_path": "/ve72U0s0wF545fdfdADFfdf.jpg"
    },
    {
        "id": 1891,
        "title": "The Empire Strikes Back",
        "release_date": date(1980, 5, 21),
        "director": "Irvin Kershner",
        "cast_members": ["Mark Hamill", "Harrison Ford", "Carrie Fisher", "Billy Dee Williams"],
        "genres": ["Sci-Fi", "Adventure", "Action"],
        "overview": "The epic space saga continues as the Empire tracks the Rebels to the ice planet Hoth, Luke trains with Jedi Master Yoda, and Darth Vader reveals a dark truth.",
        "runtime": 124,
        "vote_average": 8.4,
        "popularity": 65.50,
        "poster_path": "/7Bu4vHnEgIiU5tCfsfsaf4as.jpg"
    },
    {
        "id": 103,
        "title": "Dark City",
        "release_date": date(1998, 2, 27),
        "director": "Alex Proyas",
        "cast_members": ["Rufus Sewell", "Kiefer Sutherland", "Jennifer Connelly", "William Hurt"],
        "genres": ["Sci-Fi", "Mystery", "Thriller"],
        "overview": "A man struggles with memories of his past—including a wife he cannot remember—in a dark, nightmarish city ruled by mysterious beings who stop time and alter reality.",
        "runtime": 100,
        "vote_average": 7.3,
        "popularity": 22.40,
        "poster_path": "/fdfasdaefafdae.jpg"
    },
    {
        "id": 155,
        "title": "The Dark Knight",
        "release_date": date(2008, 7, 16),
        "director": "Christopher Nolan",
        "cast_members": ["Christian Bale", "Heath Ledger", "Aaron Eckhart", "Maggie Gyllenhaal"],
        "genres": ["Action", "Crime", "Drama"],
        "overview": "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker.",
        "runtime": 152,
        "vote_average": 8.5,
        "popularity": 135.20,
        "poster_path": "/qJ2t4QN7Zzd8ZUIco765adsa.jpg"
    },
    {
        "id": 346,
        "title": "Seven Samurai",
        "release_date": date(1954, 4, 26),
        "director": "Akira Kurosawa",
        "cast_members": ["Toshiro Mifune", "Takashi Shimura", "Yoshio Inaba"],
        "genres": ["Action", "Drama"],
        "overview": "A veteran samurai, who has fallen on hard times, answers a request for protection from a village of farmers who are repeatedly raided by bandits. He gathers six other samurai to help him instruct the villagers in self-defense.",
        "runtime": 207,
        "vote_average": 8.5,
        "popularity": 32.10,
        "poster_path": "/samurai123.jpg"
    },
    {
        "id": 769,
        "title": "GoodFellas",
        "release_date": date(1990, 9, 12),
        "director": "Martin Scorsese",
        "cast_members": ["Robert De Niro", "Ray Liotta", "Joe Pesci", "Lorraine Bracco"],
        "genres": ["Crime", "Drama"],
        "overview": "The true story of Henry Hill, his life in the mob, and his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito in the Italian-American crime syndicate.",
        "runtime": 145,
        "vote_average": 8.5,
        "popularity": 85.10,
        "poster_path": "/goodfellas.jpg"
    }
]

def seed_db():
    print("--- Starting Database Seeding ---")
    
    # 0. Pre-check: Ensure target database 'cinephile_db' exists
    try:
        from sqlalchemy import create_engine, text
        # Parse settings URL and point to default 'postgres' system database
        db_url = settings.DATABASE_URL
        postgres_url = db_url.rsplit('/', 1)[0] + '/postgres'
        
        temp_engine = create_engine(postgres_url, isolation_level="AUTOCOMMIT")
        with temp_engine.connect() as conn:
            db_exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname='cinephile_db'")).fetchone()
            if not db_exists:
                print("Database 'cinephile_db' does not exist. Creating it now...")
                conn.execute(text("CREATE DATABASE cinephile_db"))
                print("Database 'cinephile_db' created successfully.")
            else:
                print("Database 'cinephile_db' already exists.")
        temp_engine.dispose()
    except Exception as e:
        print(f"[Warning] Pre-connection database check skipped: {e}. Attempting direct connection...")

    # 1. Initialize PostgreSQL connection
    try:
        print("Connecting to PostgreSQL and creating tables...")
        Base.metadata.create_all(bind=engine)
        db: Session = SessionLocal()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to PostgreSQL database: {e}")
        print("Please ensure PostgreSQL is running locally, and your configuration details are set in the backend/.env file.")
        print("Aborting database seed.")
        sys.exit(1)

    # 2. Insert movies into PostgreSQL
    movies_to_upsert = []
    try:
        print(f"Adding {len(SEED_MOVIES)} movies to relational database...")
        for m_data in SEED_MOVIES:
            m_data.pop("poster_path", None)
            existing = db.query(Movie).filter(Movie.id == m_data["id"]).first()
            if not existing:
                new_movie = Movie(**m_data)
                db.add(new_movie)
                movies_to_upsert.append(new_movie)
                print(f"  Added: {m_data['title']}")
            else:
                movies_to_upsert.append(existing)
                print(f"  Already exists in Postgres: {m_data['title']}")
        db.commit()
        print("PostgreSQL sync complete.")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Database transaction failed during SQL seeding: {e}")
        db.close()
        sys.exit(1)

    # 3. Seed Qdrant Client (Local folder vector index)
    try:
        print("\nConnecting to Qdrant Local Client and indexing vectors...")
        print("Note: This will download sentence-transformers/all-MiniLM-L6-v2 on first run...")
        upsert_movies_to_qdrant(get_qdrant(), movies_to_upsert)
        print("Qdrant local vector database sync complete!")
    except Exception as e:
        print(f"[ERROR] Qdrant seeding failed: {e}")
        db.close()
        sys.exit(1)
        
    db.close()
    print("\n--- Seeding Completed Successfully! ---")

if __name__ == "__main__":
    seed_db()
