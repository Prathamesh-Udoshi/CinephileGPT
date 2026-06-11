import json
import google.generativeai as genai
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.memory import UserProfile, ChatMessage
from app.models.movie import Movie
from typing import Generator, List

CINEPHILE_SYSTEM_INSTRUCTIONS = """You are CinephileGPT, a passionate, opinionated, movie-obsessed conversational AI. 
Your entire existence revolves around cinema, actors, directors, genres, film history, and trivia. You speak like a dedicated film buff and cinephile who has watched every film from silent classics to modern blockbusters. 

Core Rules:
1. Be highly knowledgeable: Cite director cuts, cinematography styles (e.g., anamorphic lenses, long takes), writers, and release years where appropriate.
2. Use movie metaphors: Explain real-world situations with film references (e.g., "This project structure is more cluttered than the narrative of Spider-Man 3").
3. Be witty and humorous: Express opinionated movie takes (e.g., light-hearted jabs at CGI bloat, debates over Hitchcock vs. Welles).
4. Strictly refuse non-movie questions: If a user asks about programming, math, cooking, or general non-cinema subjects, you must refuse in character, comparing their request to a bad cinematic trope, and guide them back to movies.
5. Format: Use Markdown. Make recommendations visually appealing with clear bullet points. Include the year of release in parentheses, e.g., Pulp Fiction (1994).
"""

REFUSAL_PROMPT = """The user has asked a query unrelated to movies. 
Generate a witty, humorous, movie-obsessed refusal. Explain that you only discuss cinema. 
Compare their request to a boring, generic, or bad movie trope (e.g., 'This coding question feels like the screenplay for a straight-to-DVD sequel nobody wanted'), and redirect them to discuss films. Keep it short (1-3 sentences) and highly entertaining.

User Message: {message}"""

def get_refusal_stream(message: str) -> Generator[str, None, None]:
    """
    Stream a humorous cinephile refusal response.
    """
    if not settings.GEMINI_API_KEY:
        yield "I only talk about movies! (Configure GEMINI_API_KEY in your .env file to unlock my full cinematic wit.)"
        return
        
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            settings.GEMINI_MODEL_NAME,
            system_instruction=CINEPHILE_SYSTEM_INSTRUCTIONS
        )
        response = model.generate_content(
            REFUSAL_PROMPT.format(message=message),
            stream=True
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Look, kid, that's not a movie question. Let's talk cinema! (Error: {str(e)})"

def format_context_movies(movies: List[Movie]) -> str:
    """
    Format the retrieved RAG movie metadata into a clean XML context string.
    """
    if not movies:
        return "No specific movie metadata found in search database."
        
    context_blocks = []
    for idx, m in enumerate(movies):
        cast_str = ", ".join(m.cast_members) if m.cast_members else "Unknown"
        genres_str = ", ".join(m.genres) if m.genres else "Unknown"
        block = (
            f"--- Candidate Movie {idx+1} ---\n"
            f"ID: {m.id}\n"
            f"Title: {m.title} ({m.release_date.year if m.release_date else 'N/A'})\n"
            f"Director: {m.director or 'Unknown'}\n"
            f"Cast: {cast_str}\n"
            f"Genres: {genres_str}\n"
            f"Runtime: {m.runtime or 'Unknown'} minutes\n"
            f"Rating: {m.vote_average or 'N/A'}/10\n"
            f"Overview: {m.overview or 'No synopsis available.'}\n"
        )
        context_blocks.append(block)
    return "\n".join(context_blocks)

def get_chat_stream(
    message: str,
    history: List[ChatMessage],
    user_profile: UserProfile,
    retrieved_movies: List[Movie]
) -> Generator[str, None, None]:
    """
    Generate a personalized streaming chat response using Gemini API.
    Injects context of retrieved movies, user preferences, and chat history.
    """
    if not settings.GEMINI_API_KEY:
        yield "Gemini API Key is missing. Please add GEMINI_API_KEY to your .env file to enable conversation."
        return

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            settings.GEMINI_MODEL_NAME,
            system_instruction=CINEPHILE_SYSTEM_INSTRUCTIONS
        )
        
        # 1. Format User profile memory
        profile_str = "No specific preferences recorded yet."
        if user_profile:
            profile_str = (
                f"Favorite Genres: {user_profile.favorite_genres or []}\n"
                f"Favorite Directors: {user_profile.favorite_directors or []}\n"
                f"Favorite Actors: {user_profile.favorite_actors or []}\n"
                f"Disliked Genres: {user_profile.disliked_genres or []}\n"
                f"General Persona Notes: {user_profile.general_notes or ''}"
            )
            
        # 2. Format short term history
        history_list = []
        for msg in history:
            role_tag = "User" if msg.role == "user" else "CinephileGPT"
            history_list.append(f"{role_tag}: {msg.content}")
        history_str = "\n".join(history_list)
        
        # 3. Format RAG results
        retrieved_movies_str = format_context_movies(retrieved_movies)
        
        # 4. Construct instruction prompt
        full_prompt = f"""You are chatting with a user. Use their profile preferences to guide your response styling, recommendations, and reviews.
For movie recommendations, prioritize movies from the "RETRIEVED MOVIE DATABASE CONTEXT" below. If they ask for something outside the context, draw from your own general knowledge, but reference the context whenever possible.

USER PROFILE MEMORY:
{profile_str}

RETRIEVED MOVIE DATABASE CONTEXT (Use these for specific recommendations):
{retrieved_movies_str}

RECENT CONVERSATION HISTORY:
{history_str}

Current Message:
User: {message}

CinephileGPT response:"""

        response = model.generate_content(full_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"Cut! We had a production error: {str(e)}. Let's retake from the top."

def extract_and_consolidate_memory(
    db: Session,
    user_id: str,
    user_message: str,
    assistant_response: str
):
    """
    Run an asynchronous preference extraction prompt using Gemini API.
    Reads current user profile, parses dialogue turn, and merges new preferences.
    """
    if not settings.GEMINI_API_KEY:
        return

    try:
        # Fetch current profile or create a default one
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id, favorite_genres=[], favorite_directors=[], favorite_actors=[], disliked_genres=[], general_notes="")
            db.add(profile)
            db.commit()
            db.refresh(profile)

        # Prompt Gemini to extract explicit likes/dislikes
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)

        extraction_prompt = f"""Analyze the dialogue turn below between User and CinephileGPT. 
Extract any explicit preferences regarding movies, genres, directors, actors, or general styling.

Existing Profile:
- Favorite Genres: {profile.favorite_genres}
- Favorite Directors: {profile.favorite_directors}
- Favorite Actors: {profile.favorite_actors}
- Disliked Genres: {profile.disliked_genres}
- General Notes: {profile.general_notes}

Dialogue Turn:
User: {user_message}
CinephileGPT: {assistant_response}

Generate an updated profile incorporating any newly discovered items. Do NOT delete existing preferences unless the user explicitly contradicts them (e.g. "Actually I hate Tarantino now"). Output ONLY a JSON block with these keys: "favorite_genres", "favorite_directors", "favorite_actors", "disliked_genres", "general_notes".

Updated JSON:"""

        response = model.generate_content(
            extraction_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        updated_data = json.loads(response.text.strip())
        
        # Merge lists to prevent duplicates and maintain history
        profile.favorite_genres = list(set(profile.favorite_genres + updated_data.get("favorite_genres", [])))
        profile.favorite_directors = list(set(profile.favorite_directors + updated_data.get("favorite_directors", [])))
        profile.favorite_actors = list(set(profile.favorite_actors + updated_data.get("favorite_actors", [])))
        profile.disliked_genres = list(set(profile.disliked_genres + updated_data.get("disliked_genres", [])))
        
        new_notes = updated_data.get("general_notes", "")
        if new_notes and new_notes != profile.general_notes:
            if profile.general_notes:
                profile.general_notes = f"{profile.general_notes}\n{new_notes}".strip()
            else:
                profile.general_notes = new_notes
                
        db.commit()
        
    except Exception as e:
        print(f"Error during memory consolidation: {e}")
