import json
import httpx
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
6. Brevity is key: Keep your responses concise, sharp, and focused. Avoid overly long explanations or rambling essays. Keep the total response length short.
"""

GEMINI_SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

REFUSAL_PROMPT = """The user has asked a query unrelated to movies. 
Generate a witty, humorous, movie-obsessed refusal. Explain that you only discuss cinema. 
Compare their request to a boring, generic, or bad movie trope (e.g., 'This coding question feels like the screenplay for a straight-to-DVD sequel nobody wanted'), and redirect them to discuss films. Keep it extremely brief (maximum 1 or 2 sentences) and highly entertaining. Do NOT write more than two sentences.

User Message: {message}"""

def get_groq_completions_stream(prompt: str, system_instruction: str) -> Generator[str, None, None]:
    """
    Helper function to stream chat completion chunks from Groq API.
    """
    if not settings.GROQ_API_KEY:
        yield "\n[Error: Groq API key is missing. Cannot perform fallback.]"
        return

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.GROQ_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "temperature": 0.7
    }

    try:
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=30.0) as r:
            if r.status_code != 200:
                error_body = r.read().decode("utf-8")
                yield f"\n[Warning: Groq Fallback failed with HTTP {r.status_code}: {error_body}]"
                return
                
            for line in r.iter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue
    except Exception as e:
        yield f"\n[Warning: Groq Connection error: {str(e)}]"

def clean_stream_prefixes(stream: Generator[str, None, None]) -> Generator[str, None, None]:
    """
    Buffer the start of a stream and clean up common model-generated prefixes 
    like 'assistant:', 'assistant\n', 'CinephileGPT:', or 'CinephileGPT response:'.
    """
    buffer = ""
    prefix_checked = False
    
    for chunk in stream:
        if not prefix_checked:
            buffer += chunk
            # Wait until we have enough characters or a newline/colon to check for prefixes
            if len(buffer) >= 40 or "\n" in chunk or ":" in chunk:
                cleaned_buffer = buffer
                lower_buf = cleaned_buffer.lower().strip()
                
                prefixes_to_strip = [
                    "assistant:",
                    "assistant\n",
                    "assistant",
                    "cinephilegpt:",
                    "cinephilegpt response:",
                    "cinephilegpt\n"
                ]
                
                stripped = True
                while stripped:
                    stripped = False
                    for prefix in prefixes_to_strip:
                        val = cleaned_buffer.lstrip()
                        if val.lower().startswith(prefix):
                            cleaned_buffer = val[len(prefix):].lstrip()
                            stripped = True
                            break
                            
                if cleaned_buffer:
                    yield cleaned_buffer
                buffer = ""
                prefix_checked = True
        else:
            yield chunk
            
    if not prefix_checked and buffer:
        cleaned_buffer = buffer
        prefixes_to_strip = [
            "assistant:",
            "assistant\n",
            "assistant",
            "cinephilegpt:",
            "cinephilegpt response:",
            "cinephilegpt\n"
        ]
        
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes_to_strip:
                val = cleaned_buffer.lstrip()
                if val.lower().startswith(prefix):
                    cleaned_buffer = val[len(prefix):].lstrip()
                    stripped = True
                    break
        if cleaned_buffer:
            yield cleaned_buffer

def _get_refusal_stream_raw(message: str) -> Generator[str, None, None]:
    """
    Stream a humorous cinephile refusal response.
    Tries Gemini API first, falling back to Groq API on failure.
    """
    prompt = REFUSAL_PROMPT.format(message=message)

    # 1. Try Gemini
    if settings.GEMINI_API_KEY:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(
                settings.GEMINI_MODEL_NAME,
                system_instruction=CINEPHILE_SYSTEM_INSTRUCTIONS,
                safety_settings=GEMINI_SAFETY_SETTINGS
            )
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                try:
                    text = chunk.text
                except (ValueError, IndexError) as error:
                    raise ValueError(f"Empty or blocked response chunk ({error})")
                if text:
                    yield text
            return # Succeeded
        except Exception as e:
            print(f"[Warning] Gemini refusal stream failed: {e}. Falling back to Groq...")

    # 2. Try Groq Fallback
    if settings.GROQ_API_KEY:
        try:
            for chunk in get_groq_completions_stream(prompt, CINEPHILE_SYSTEM_INSTRUCTIONS):
                yield chunk
            return # Succeeded
        except Exception as e:
            print(f"[Error] Groq fallback failed: {e}")

    yield "I only talk about movies! (Both Gemini and Groq API backends encountered rate limits or connection errors.)"

def get_refusal_stream(message: str) -> Generator[str, None, None]:
    return clean_stream_prefixes(_get_refusal_stream_raw(message))

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

def _get_chat_stream_raw(
    message: str,
    history: List[ChatMessage],
    user_profile: UserProfile,
    retrieved_movies: List[Movie],
    intent: str = "MOVIE_DISCUSSION"
) -> Generator[str, None, None]:
    """
    Generate a personalized streaming chat response using Gemini or Groq API.
    Injects context of retrieved movies, user preferences, and chat history.
    """
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
    recommendation_constraint = ""
    if intent == "MOVIE_RECOMMENDATION":
        recommendation_constraint = """
CRITICAL RECOMMENDATION CONSTRAINT:
1. FULFILL requested criteria: If the user explicitly asks for a specific genre, style, director, or era (e.g., "Suggest some horror movies"), you MUST recommend films matching that request directly.
2. NEVER mention or lecture about the user profile: Do NOT mention the user's profile preferences, do not point out that their current request deviates from their profile, and do not compare their request to their profile (e.g., do NOT say "It seems you are taking a detour from your usual..."). Fulfill their query directly as if they are asking for it normally.
3. Database vs. General Knowledge: Prioritize recommending movies from the "RETRIEVED MOVIE DATABASE CONTEXT" below. However, if the context has no matching movies, you MUST recommend matching movies from your own general knowledge. Do NOT explain or apologize that the database doesn't have the movies; just list the recommendations seamlessly.
4. Format: Recommendations must be a clear, markdown-formatted list of movies. Each item must have the title, year of release in parentheses, and a 1-2 sentence description. E.g.:
   * **The Shining (1980)**: Description here.
5. Brevity: Keep the response extremely brief, punchy, and structured. Keep the intro and outro to at most 1 short sentence each.
"""
    else:
        recommendation_constraint = """
CRITICAL BREVITY CONSTRAINT: 
The user is NOT asking for movie recommendations. Respond meaningfully to their message in character (discussing cinema history, philosophy, metaphors, or answering their specific movie-related question), but DO NOT list, suggest, or recommend any movies to watch. Absolutely do not include any lists or bullet points of movies to watch. Keep your answer brief and to the point (maximum 1 or 2 short paragraphs total).
"""

    full_prompt = f"""You are chatting with a user. Use their profile preferences to guide your response styling, recommendations, and reviews, but remember that the user's current message always overrides their general profile preferences.

CRITICAL PERSONALIZATION & CONTEXT RULES:
1. Conversation Continuity: Always prioritize the actual topic of discussion. If the user is responding to your question or discussing a specific movie/director (e.g., you asked about Pulp Fiction, and they said "I haven't watched it yet"), focus your reply directly on that topic, director (Quentin Tarantino), or genre.
2. No Profile Referencing: Do NOT mention or refer to the user's stored profile favorites directly in the conversation. For example, do NOT say "I know you like Brad Pitt", "Since you are a fan of Christopher Nolan", or "It's no surprise you love psychological thrillers." Use these preferences subtly behind the scenes to guide your recommendations and style, behaving as if you simply happen to share similar tastes, without revealing that you are tracking their profile database metadata.
3. No Shoehorning: Do NOT force the user's profile favorites (like David Fincher, Morgan Freeman, psychological thrillers) into the conversation unless they are directly relevant to the current topic. The profile memory is a subtle guide, not a mandate to repeat their favorites in every message.
4. Independent Cinephile Identity: When the user asks about YOUR (the assistant's) personal favorites, preferences, or opinions (e.g. "who is your favorite actor?", "what is your favorite movie?", "which director do you like?"), express your own independent, opinionated cinephile views from general cinema history. Do NOT copy, parrot, or feed back the user's own profile favorites (from the USER PROFILE MEMORY) as if they are your own. Be a distinct entity with your own unique cinematic taste.
5. Natural Flow: Address the user's current sentence/message directly and keep the conversation natural, engaging, and in character as CinephileGPT, without sounding robotic or stalker-like.

{recommendation_constraint}
For movie recommendations, prioritize movies from the "RETRIEVED MOVIE DATABASE CONTEXT" below. If they ask for something outside the context, draw from your own general knowledge, but reference the context whenever possible. You must never mention database limits, database search contexts, or user profile detours to the user.

USER PROFILE MEMORY:
{profile_str}

RETRIEVED MOVIE DATABASE CONTEXT (Use these for specific recommendations):
{retrieved_movies_str}

RECENT CONVERSATION HISTORY:
{history_str}

Current Message:
User: {message}

CinephileGPT response:"""

    # 1. Try Gemini first
    if settings.GEMINI_API_KEY:
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(
                settings.GEMINI_MODEL_NAME,
                system_instruction=CINEPHILE_SYSTEM_INSTRUCTIONS,
                safety_settings=GEMINI_SAFETY_SETTINGS
            )
            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                try:
                    text = chunk.text
                except (ValueError, IndexError) as error:
                    raise ValueError(f"Empty or blocked response chunk ({error})")
                if text:
                    yield text
            return # Succeeded
        except Exception as e:
            print(f"[Warning] Gemini chat stream failed: {e}. Falling back to Groq...")

    # 2. Try Groq Fallback
    if settings.GROQ_API_KEY:
        try:
            for chunk in get_groq_completions_stream(full_prompt, CINEPHILE_SYSTEM_INSTRUCTIONS):
                yield chunk
            return # Succeeded
        except Exception as e:
            print(f"[Error] Groq chat stream fallback failed: {e}")

    yield "Look, kid, we had a production error (both Gemini and Groq APIs are currently offline or overloaded). Let's retake from the top."

def get_chat_stream(
    message: str,
    history: List[ChatMessage],
    user_profile: UserProfile,
    retrieved_movies: List[Movie],
    intent: str = "MOVIE_DISCUSSION"
) -> Generator[str, None, None]:
    return clean_stream_prefixes(_get_chat_stream_raw(message, history, user_profile, retrieved_movies, intent))

def extract_and_consolidate_memory(
    db: Session,
    user_id: str,
    user_message: str,
    assistant_response: str
):
    """
    Run an asynchronous preference extraction prompt using Gemini API (with Groq fallback).
    Reads current user profile, parses dialogue turn, and merges new preferences.
    """
    try:
        # Fetch current profile or create a default one
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id, favorite_genres=[], favorite_directors=[], favorite_actors=[], disliked_genres=[], general_notes="")
            db.add(profile)
            db.commit()
            db.refresh(profile)

        # Build extraction prompt
        extraction_prompt = f"""You are an analytical backend worker. Your task is to update a user's movie preference profile based on a new dialogue turn.

CRITICAL RULES:
1. Extract preferences explicitly stated or agreed to by the **User** (e.g., "I hate romance", "Fincher is my favorite"). Additionally, if the user expresses love for a specific movie (e.g. "I love Seven"), you SHOULD extract its main director (e.g., David Fincher) and main actors (e.g., Brad Pitt, Morgan Freeman) into the favorite directors and favorite actors lists.
2. DO NOT extract or assume preferences from suggestions, recommendations, or movies/directors mentioned ONLY by **CinephileGPT** (the assistant) unless the User explicitly agrees to them. However, if the assistant mentions the director or actors of a movie the user likes (e.g. CinephileGPT mentions that David Fincher directed Seven, which the user liked), you can safely extract them as favorites.
3. Keep the lists clean, normalized, and unique.
4. For "general_notes", consolidate the existing notes with any new observations from the User's message into a single, cohesive, plain-text paragraph summarizing the user's overall cinematic tastes and style preferences. Do NOT write list formatting (like brackets, quotes, sets, or braces) inside "general_notes".

Existing Profile:
- Favorite Genres: {profile.favorite_genres}
- Favorite Directors: {profile.favorite_directors}
- Favorite Actors: {profile.favorite_actors}
- Disliked Genres: {profile.disliked_genres}
- General Notes: {profile.general_notes}

Dialogue Turn:
User: {user_message}
CinephileGPT: {assistant_response}

Generate the updated profile incorporating any newly discovered items from the User's message. Output ONLY a valid JSON block with these keys: "favorite_genres", "favorite_directors", "favorite_actors", "disliked_genres", "general_notes".

Updated JSON:"""

        updated_data = None

        # 1. Try Gemini
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    settings.GEMINI_MODEL_NAME,
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
                response = model.generate_content(
                    extraction_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                try:
                    text = response.text
                except (ValueError, IndexError) as error:
                    raise ValueError(f"Empty or blocked response ({error})")
                updated_data = json.loads(text.strip())
            except Exception as e:
                print(f"[Warning] Gemini memory extraction failed: {e}. Trying Groq fallback...")

        # 2. Try Groq
        if not updated_data and settings.GROQ_API_KEY:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": settings.GROQ_MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2
                }
                r = httpx.post(url, headers=headers, json=payload, timeout=20.0)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    updated_data = json.loads(content.strip())
                else:
                    print(f"[Warning] Groq memory API returned {r.status_code}: {r.text}")
            except Exception as e:
                print(f"[Error] Groq memory extraction failed: {e}")

        # Save updates to DB if either LLM responded
        if updated_data:
            profile.favorite_genres = list(set(profile.favorite_genres + updated_data.get("favorite_genres", [])))
            profile.favorite_directors = list(set(profile.favorite_directors + updated_data.get("favorite_directors", [])))
            profile.favorite_actors = list(set(profile.favorite_actors + updated_data.get("favorite_actors", [])))
            profile.disliked_genres = list(set(profile.disliked_genres + updated_data.get("disliked_genres", [])))
            
            new_notes = updated_data.get("general_notes", "")
            if new_notes:
                profile.general_notes = new_notes.strip()
                
            db.commit()
            
    except Exception as e:
        print(f"Error during memory consolidation: {e}")
