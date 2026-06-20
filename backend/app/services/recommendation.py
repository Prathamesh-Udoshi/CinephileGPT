import json
import logging
from typing import List, Tuple, Optional, Generator
from uuid import UUID
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
import google.generativeai as genai
import httpx

from app.core.config import settings
from app.models.memory import UserProfile, ChatMessage, RecommendationLog
from app.models.movie import Movie, UserWatchlist
from app.services.retrieval import search_movies_vector
from app.services.llm import CINEPHILE_SYSTEM_INSTRUCTIONS, GEMINI_SAFETY_SETTINGS, get_groq_completions_stream

class RecommendationPipeline:
    logger = logging.getLogger("recommendation_pipeline")

    @classmethod
    def detect_intent(cls, message: str) -> str:
        """
        Detects query intent. Returns classification.
        """
        from app.services.intent import IntentClassifierService
        return IntentClassifierService.classify_intent(message)

    @classmethod
    def _get_json_completion(cls, prompt: str) -> dict:
        """
        Get JSON output from LLM (Gemini or Groq fallback).
        """
        # 1. Try Gemini
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    settings.GEMINI_MODEL_NAME,
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text.strip())
            except Exception as e:
                cls.logger.warning(f"Gemini JSON completion failed: {e}. Trying Groq fallback...")
                
        # 2. Try Groq Fallback
        if settings.GROQ_API_KEY:
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
                    "temperature": 0.2
                }
                r = httpx.post(url, headers=headers, json=payload, timeout=20.0)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    return json.loads(content.strip())
            except Exception as e:
                cls.logger.error(f"Groq JSON completion failed: {e}")
                
        # Default empty profile
        return {
            "is_complete": True, 
            "reason": "No API keys", 
            "extracted_profile": {}, 
            "follow_up_question": "What kind of movie would you like to watch?"
        }

    @classmethod
    def _get_text_completion(cls, prompt: str) -> str:
        """
        Get text completion from LLM.
        """
        # 1. Try Gemini
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    settings.GEMINI_MODEL_NAME,
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                cls.logger.warning(f"Gemini text completion failed: {e}. Trying Groq...")
                
        # 2. Try Groq
        if settings.GROQ_API_KEY:
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
                    "temperature": 0.2
                }
                r = httpx.post(url, headers=headers, json=payload, timeout=20.0)
                if r.status_code == 200:
                    data = r.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                cls.logger.error(f"Groq text completion failed: {e}")
                
        return "A good movie recommendation"

    @classmethod
    def _get_stream_completion(cls, prompt: str, llm_info: dict = None) -> Generator[str, None, None]:
        """
        Stream text completion from LLM.
        """
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
                first = True
                for chunk in response:
                    try:
                        text = chunk.text
                    except (ValueError, IndexError) as error:
                        raise ValueError(f"Empty or blocked response chunk ({error})")
                    if text:
                        if first and llm_info is not None:
                            llm_info["provider"] = "Gemini"
                            first = False
                        yield text
                return
            except Exception as e:
                cls.logger.warning(f"Gemini stream failed: {e}. Trying Groq fallback...")
                
        # 2. Try Groq
        if settings.GROQ_API_KEY:
            try:
                first = True
                for chunk in get_groq_completions_stream(prompt, CINEPHILE_SYSTEM_INSTRUCTIONS):
                    if first and llm_info is not None:
                        llm_info["provider"] = "Groq"
                        first = False
                    yield chunk
                return
            except Exception as e:
                cls.logger.error(f"Groq stream failed: {e}")
                
        if llm_info is not None:
            llm_info["provider"] = "None"
        yield "Look, kid, we had a production error. Let's take it from the top."

    @classmethod
    def assess_completeness(
        cls,
        user_id: str,
        message: str,
        history: List[ChatMessage],
        db: Session,
        is_evaluation: bool = False
    ) -> Tuple[bool, dict, Optional[str]]:
        """
        Evaluates completeness of user request preferences.
        """
        # Fetch profile
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        
        # Serialize history
        history_list = []
        for msg in history:
            role = "User" if msg.role == "user" else "CinephileGPT"
            history_list.append(f"{role}: {msg.content}")
        history_str = "\n".join(history_list)

        prompt = f"""You are the Information Completeness Evaluator for CinephileGPT, an expert movie concierge system.
Your job is to assess if we have enough details to retrieve high-quality, personalized movie recommendations for the user.

Stated user message: "{message}"

Recent conversation history:
{history_str}

Historical preferences:
- Favorite Genres: {profile.favorite_genres if profile else []}
- Favorite Directors: {profile.favorite_directors if profile else []}
- Favorite Actors: {profile.favorite_actors if profile else []}
- Disliked Genres: {profile.disliked_genres if profile else []}
- General Notes: {profile.general_notes if profile else ""}

Check if the current message, the conversation history, or historical preferences provide specific search parameters.
Relevant attributes: genre, mood, themes, pacing (slow burn vs fast paced), language, country, release period, runtime, favorite actors/directors/movies, mainstream vs hidden gem, emotional impact.

CRITICAL RULES:
1. Information is COMPLETE if the user specifies at least one constraint (genre, mood, theme, director, actor, similar movie, pacing, etc.) in the query or history.
2. Information is INCOMPLETE if the query is very generic (e.g. "Recommend a movie", "Suggest something to watch", "What should I watch tonight?") and no other details/history are present.
3. If historical preferences exist but the query is generic, acknowledge that memory and ask a follow-up question (e.g. 'I know you consistently enjoy science fiction. Do you want to stick with that tonight or try something different?') and set is_complete to false.
4. If is_evaluation is true, ALWAYS set is_complete to true if there is any historical preference or query constraint, so that we can immediately recommend a movie.

Is Evaluation Run: {is_evaluation}

Return ONLY a valid JSON object:
{{
  "is_complete": true/false,
  "reason": "...",
  "extracted_profile": {{
    "genres": [...],
    "moods": [...],
    "themes": [...],
    "pacing": "slow burn" or "fast-paced" or null,
    "languages": [...],
    "release_period": "post-2015" or null,
    "runtime_limit_minutes": integer or null,
    "favorite_directors": [...],
    "favorite_actors": [...],
    "favorite_movies": [...],
    "exclude_genres": [...],
    "exclude_movie_titles": [...],
    "mainstream_vs_hidden_gem": "mainstream" or "hidden_gem" or null
  }},
  "follow_up_question": "..."
}}
"""
        result = cls._get_json_completion(prompt)
        is_complete = result.get("is_complete", True)
        extracted = result.get("extracted_profile", {})
        follow_up = result.get("follow_up_question")
        
        # Double safety check: if we are in evaluation and have any memory/extraction, force complete
        if is_evaluation:
            # If user has some history preference, or we parsed genres/directors/actors/themes, set is_complete to True
            has_extracted_data = any(extracted.get(k) for k in extracted if extracted.get(k))
            has_profile_data = profile and (profile.favorite_genres or profile.favorite_directors or profile.favorite_actors)
            if has_extracted_data or has_profile_data or len(message) > 15:
                is_complete = True
                follow_up = None

        return is_complete, extracted, follow_up

    @classmethod
    def build_preference_profile(
        cls,
        user_id: str,
        extracted_profile: dict,
        db: Session
    ) -> dict:
        """
        Merges long-term PostgreSQL profile settings, watchlists, exclusions and new conversation preferences.
        """
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        
        genres = list(set(extracted_profile.get("genres", []) + (profile.favorite_genres if profile else [])))
        directors = list(set(extracted_profile.get("favorite_directors", []) + (profile.favorite_directors if profile else [])))
        actors = list(set(extracted_profile.get("favorite_actors", []) + (profile.favorite_actors if profile else [])))
        exclude_genres = list(set(extracted_profile.get("exclude_genres", []) + (profile.disliked_genres if profile else [])))
        
        for g in genres:
            if g in exclude_genres:
                exclude_genres.remove(g)
                
        watchlist_items = db.query(UserWatchlist).filter(
            UserWatchlist.user_id == user_id,
            UserWatchlist.status.in_(["watched", "disliked"])
        ).all()
        watched_movie_ids = [item.movie_id for item in watchlist_items]
        
        recent_logs = db.query(RecommendationLog).filter(
            RecommendationLog.user_id == user_id
        ).order_by(RecommendationLog.timestamp.desc()).limit(15).all()
        
        previously_recommended_ids = []
        for log in recent_logs:
            if log.recommended_movie_ids:
                previously_recommended_ids.extend(log.recommended_movie_ids)
        previously_recommended_ids = list(set(previously_recommended_ids))
        
        merged = {
            "genres": genres,
            "moods": extracted_profile.get("moods", []),
            "themes": extracted_profile.get("themes", []),
            "pacing": extracted_profile.get("pacing"),
            "languages": extracted_profile.get("languages", []),
            "release_period": extracted_profile.get("release_period"),
            "runtime_limit_minutes": extracted_profile.get("runtime_limit_minutes"),
            "favorite_directors": directors,
            "favorite_actors": actors,
            "favorite_movies": extracted_profile.get("favorite_movies", []),
            "exclude_genres": exclude_genres,
            "exclude_movie_ids": list(set(watched_movie_ids + previously_recommended_ids)),
            "mainstream_vs_hidden_gem": extracted_profile.get("mainstream_vs_hidden_gem")
        }
        return merged

    @classmethod
    def generate_retrieval_query(cls, profile: dict) -> str:
        """
        Converts the preference profile into a semantically descriptive search query.
        """
        summary = (
            f"Genres: {profile.get('genres')}\n"
            f"Moods: {profile.get('moods')}\n"
            f"Themes: {profile.get('themes')}\n"
            f"Pacing: {profile.get('pacing')}\n"
            f"Mainstream vs Hidden Gem: {profile.get('mainstream_vs_hidden_gem')}\n"
            f"Favorite Directors: {profile.get('favorite_directors')}\n"
            f"Favorite Actors: {profile.get('favorite_actors')}\n"
        )
        prompt = f"""You are a movie retrieval query generator.
Based on the following user preference profile, generate a single, semantically rich retrieval query for a vector database search.
The query should describe the desired film's tone, atmosphere, genres, themes, and pacing. Do NOT output keywords or instructions.
Example: "A tense, slow-burn psychological thriller exploring themes of obsession and identity with strong cinematography"

Preference Profile:
{summary}

Write ONLY the generated search query.
Generated Query:"""
        return cls._get_text_completion(prompt)

    @classmethod
    def hybrid_retrieve(
        cls,
        query: str,
        profile: dict,
        db: Session,
        client: QdrantClient,
        extracted_profile: dict = None,
        original_query: str = None
    ) -> List[Movie]:
        """
        Fetches candidates from Qdrant, applies PostgreSQL watchlists/logs exclusions and soft filter limits.
        """
        # Fetch top 50 candidates from Qdrant
        vector_results = search_movies_vector(
            client=client,
            query=query,
            limit=50,
            favorite_genres=None,
            disliked_genres=profile.get("exclude_genres")
        )
        
        if not vector_results:
            query_db = db.query(Movie)
            if profile.get("exclude_genres"):
                for genre in profile.get("exclude_genres"):
                    query_db = query_db.filter(~Movie.genres.any(genre))
            candidates = query_db.order_by(Movie.popularity.desc()).limit(20).all()
        else:
            movie_ids = [res["movie_id"] for res in vector_results]
            candidates = db.query(Movie).filter(Movie.id.in_(movie_ids)).all()
            movie_dict = {m.id: m for m in candidates}
            candidates = [movie_dict[m_id] for m_id in movie_ids if m_id in movie_dict]

        # Explicit preference boosting: If director/actor/movies are explicitly in the current query,
        # query the SQL database directly and boost them to the top of the candidates list
        explicit_directors = list(extracted_profile.get("favorite_directors", [])) if extracted_profile else []
        search_text = (original_query or query or "").lower()
        
        if not explicit_directors and profile.get("favorite_directors"):
            for d in profile.get("favorite_directors", []):
                if d:
                    parts = d.lower().split()
                    if d.lower() in search_text or (parts and parts[-1] in search_text):
                        explicit_directors.append(d)

        # Fallback database scanning for any director matching words in the original query if none matched
        if not explicit_directors and original_query:
            clean_query = original_query.lower()
            stopwords = {"movies", "movie", "from", "by", "director", "directed", "with", "actor", "actress", "star", "starring", "in", "of", "the", "a", "an", "and", "or", "to", "show", "shows", "film", "films"}
            query_words = [w.strip(",.!?\"'()[]") for w in clean_query.split()]
            search_words = [w for w in query_words if w and w not in stopwords and len(w) > 2]
            
            for word in search_words:
                matched_directors = db.query(Movie.director).filter(Movie.director.ilike(f"%{word}%")).distinct().limit(5).all()
                for (d_name,) in matched_directors:
                    if d_name:
                        parts = d_name.lower().split()
                        if any(p == word for p in parts) or d_name.lower() in clean_query:
                            if d_name not in explicit_directors:
                                explicit_directors.append(d_name)

        # Filter explicit directors to only keep those actually mentioned in the original query
        if original_query and explicit_directors:
            clean_query = original_query.lower()
            filtered_directors = []
            for d in explicit_directors:
                if d:
                    parts = d.lower().split()
                    if d.lower() in clean_query or (parts and parts[-1] in clean_query):
                        filtered_directors.append(d)
            explicit_directors = filtered_directors

        explicit_actors = list(extracted_profile.get("favorite_actors", [])) if extracted_profile else []
        if not explicit_actors and profile.get("favorite_actors"):
            for a in profile.get("favorite_actors", []):
                if a:
                    parts = a.lower().split()
                    if a.lower() in search_text or (parts and parts[-1] in search_text):
                        explicit_actors.append(a)

        # Filter explicit actors to only keep those actually mentioned in the original query
        if original_query and explicit_actors:
            clean_query = original_query.lower()
            filtered_actors = []
            for a in explicit_actors:
                if a:
                    parts = a.lower().split()
                    if a.lower() in clean_query or (parts and parts[-1] in clean_query):
                        filtered_actors.append(a)
            explicit_actors = filtered_actors

        explicit_movies = list(extracted_profile.get("favorite_movies", [])) if extracted_profile else []
        if not explicit_movies and profile.get("favorite_movies"):
            for m in profile.get("favorite_movies", []):
                if m and m.lower() in search_text:
                    explicit_movies.append(m)

        # Filter explicit movies to only keep those actually mentioned in the original query
        if original_query and explicit_movies:
            clean_query = original_query.lower()
            filtered_movies = []
            for m in explicit_movies:
                if m and m.lower() in clean_query:
                    filtered_movies.append(m)
            explicit_movies = filtered_movies

        from sqlalchemy import or_
        conditions = []
        if explicit_directors:
            conditions.extend([Movie.director.ilike(f"%{d.strip()}%") for d in explicit_directors if d and d.strip()])
        if explicit_actors:
            conditions.extend([Movie.cast_members.any(a.strip()) for a in explicit_actors if a and a.strip()])
        if explicit_movies:
            conditions.extend([Movie.title.ilike(f"%{m.strip()}%") for m in explicit_movies if m and m.strip()])

        cls.logger.info(f"hybrid_retrieve: original_query='{original_query}', query='{query}'")
        cls.logger.info(f"hybrid_retrieve: explicit_directors={explicit_directors}, explicit_actors={explicit_actors}, explicit_movies={explicit_movies}")

        if conditions:
            db_boosted = db.query(Movie).filter(or_(*conditions)).all()
            cls.logger.info(f"hybrid_retrieve: db_boosted count={len(db_boosted)}: {[m.title for m in db_boosted]}")
            if db_boosted:
                boosted_ids = {m.id for m in db_boosted}
                candidates = db_boosted + [c for c in candidates if c.id not in boosted_ids]
            
        exclude_movie_ids = set(profile.get("exclude_movie_ids", []))
        
        filtered = []
        for m in candidates:
            # Strictly exclude watched/disliked or previously suggested movies
            if m.id in exclude_movie_ids:
                # Bypass if the title is explicitly requested
                if m.title.lower() in query.lower():
                    pass
                else:
                    continue
                
            # Runtime limit filter
            runtime_limit = profile.get("runtime_limit_minutes")
            if runtime_limit and m.runtime and m.runtime > runtime_limit:
                continue
                
            # Release period filter
            release_period = profile.get("release_period")
            if release_period and m.release_date:
                year = m.release_date.year
                release_period_str = str(release_period).lower()
                import re
                if "after" in release_period_str:
                    match = re.search(r"\d{4}", release_period_str)
                    if match and year <= int(match.group(0)):
                        continue
                elif "before" in release_period_str:
                    match = re.search(r"\d{4}", release_period_str)
                    if match and year >= int(match.group(0)):
                        continue
                elif "90s" in release_period_str or "1990" in release_period_str:
                    if not (1990 <= year <= 1999):
                        continue
                elif "80s" in release_period_str or "1980" in release_period_str:
                    if not (1980 <= year <= 1989):
                        continue
                elif "last decade" in release_period_str:
                    if year < 2016:
                        continue

            # Hidden Gem filter
            mainstream_vs_hidden_gem = profile.get("mainstream_vs_hidden_gem")
            if mainstream_vs_hidden_gem == "hidden_gem":
                if m.popularity and m.popularity > 25.0:
                    continue
            elif mainstream_vs_hidden_gem == "mainstream":
                if m.popularity and m.popularity < 15.0:
                    continue
                    
            filtered.append(m)
            
        # Fallback if too aggressive
        if len(filtered) < 3:
            filtered = []
            for m in candidates:
                if m.id in exclude_movie_ids:
                    continue
                filtered.append(m)
                
        return filtered[:5]

    @classmethod
    def stream_recommendations(
        cls,
        movies: List[Movie],
        profile: dict,
        history: List[ChatMessage],
        current_message: str,
        llm_info: dict = None
    ) -> Generator[str, None, None]:
        """
        Streams recommendations with non-spoiler explanations.
        """
        movies_str = ""
        for idx, m in enumerate(movies):
            genres_str = ", ".join(m.genres) if m.genres else "Unknown"
            movies_str += (
                f"- Candidate {idx+1}: {m.title} ({m.release_date.year if m.release_date else 'N/A'})\n"
                f"  Director: {m.director or 'Unknown'}\n"
                f"  Genres: {genres_str}\n"
                f"  Overview: {m.overview}\n"
                f"  Runtime: {m.runtime} mins\n"
                f"  Rating: {m.vote_average}/10\n"
            )
            
        profile_str = (
            f"Stated Genres: {profile.get('genres')}\n"
            f"Mood/Themes: {profile.get('moods')}, {profile.get('themes')}\n"
            f"Favorite Director/Actor: {profile.get('favorite_directors')}, {profile.get('favorite_actors')}\n"
        )
        
        history_list = []
        for msg in history:
            role = "User" if msg.role == "user" else "CinephileGPT"
            history_list.append(f"{role}: {msg.content}")
        history_str = "\n".join(history_list)
        
        prompt = f"""You are CinephileGPT, the expert movie concierge.
Below are candidate movies retrieved from our database matching the user's taste profile.

CANDIDATE MOVIES:
{movies_str}

USER PREFERENCE PROFILE:
{profile_str}

RECENT CONVERSATION HISTORY:
{history_str}

Current User Message: "{current_message}"

CRITICAL INSTRUCTIONS:
1. Recommend 2 to 3 movies from the CANDIDATE MOVIES that best match the request.
2. For each recommended movie, write a clear, personalized explanation. Reference its atmosphere, themes, storytelling style, pacing, or character dynamics without revealing spoilers.
3. DO NOT list candidate numbers (e.g. "Candidate 1"). Use the movie titles directly.
4. Format: Use a markdown bulleted list. The title must have the release year in parentheses, e.g.:
   * **Movie Title (Year)**: Explanation here.
5. Tone: Be enthusiastic, opinionated, witty, and cinephile-obsessed. Keep introduction and outro very brief.
6. Do NOT mention database limits, database searches, or user profile detours.
"""
        for chunk in cls._get_stream_completion(prompt, llm_info):
            yield chunk

    @classmethod
    def stream_follow_up_question(cls, question: str) -> Generator[str, None, None]:
        """
        Simulate streaming the follow-up question.
        """
        yield question

    @classmethod
    def log_recommendation(
        cls,
        user_id: str,
        session_id: UUID,
        profile: dict,
        query: str,
        retrieved_movies: List[Movie],
        recommended_movies: List[Movie],
        response: str,
        db: Session,
        original_query: str = None,
        cache_hit: bool = False,
        retrieval_latency_ms: float = 0.0,
        llm_provider: str = "None",
        llm_latency_ms: float = 0.0,
        total_response_time_ms: float = 0.0,
        error_message: str = None
    ):
        """
        Persistently saves recommendation details to the database and logs.
        """
        try:
            retrieved_ids = [m.id for m in retrieved_movies]
            recommended_ids = [m.id for m in recommended_movies]
            
            log_entry = RecommendationLog(
                user_id=user_id,
                session_id=session_id,
                known_information=profile,
                information_complete=True,
                generated_query=query,
                retrieved_movie_ids=retrieved_ids,
                recommended_movie_ids=recommended_ids,
                explanations={"response": response},
                original_query=original_query,
                cache_hit=cache_hit,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_provider=llm_provider,
                llm_latency_ms=llm_latency_ms,
                total_response_time_ms=total_response_time_ms,
                error_message=error_message
            )
            db.add(log_entry)
            db.commit()
            
            cls.logger.info(f"--- Recommendation Pipeline Log ---")
            cls.logger.info(f"User ID: {user_id}")
            cls.logger.info(f"Session ID: {session_id}")
            cls.logger.info(f"Preference Profile: {profile}")
            cls.logger.info(f"Generated Semantic Query: {query}")
            cls.logger.info(f"Retrieved Movies: {[m.title for m in retrieved_movies]}")
            cls.logger.info(f"Recommended Movies: {[m.title for m in recommended_movies]}")
        except Exception as e:
            cls.logger.error(f"Error logging recommendation to DB: {e}")
