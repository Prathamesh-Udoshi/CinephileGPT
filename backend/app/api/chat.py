import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse
from typing import List
from uuid import UUID

from app.core.database import get_db, get_qdrant, SessionLocal
from app.core.security import get_current_user
from app.models.user import User
from app.models.memory import ChatSession, ChatMessage, UserProfile
from app.schemas.chat import ChatSessionResponse, ChatStreamRequest, FeedbackRequest
from app.services.intent import IntentClassifierService
from app.services.retrieval import hybrid_retrieval
from app.services.llm import get_refusal_stream, get_chat_stream, extract_and_consolidate_memory
from app.services.recommendation import RecommendationPipeline
from app.services.cache import get_cache_service

router = APIRouter(prefix="/api/chat", tags=["chat"])

def background_memory_consolidate(user_id: str, user_msg: str, assistant_reply: str):
    """
    Background worker that runs memory consolidation with a fresh database session.
    """
    db = SessionLocal()
    try:
        extract_and_consolidate_memory(db, str(user_id), user_msg, assistant_reply)
    finally:
        db.close()

@router.post("/session", response_model=ChatSessionResponse)
def create_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initialize a new chat session for the user.
    """
    session = ChatSession(user_id=current_user.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/sessions", response_model=List[ChatSessionResponse])
def get_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all chat sessions belonging to the current user.
    """
    return db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()

@router.get("/session/{session_id}", response_model=ChatSessionResponse)
def get_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve message history for a specific chat session.
    """
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found."
        )
    return session

@router.post("/stream")
async def stream_chat(
    payload: ChatStreamRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    qdrant = Depends(get_qdrant),
    cache = Depends(get_cache_service)
):
    """
    Core conversational streaming route using Server-Sent Events (SSE).
    Detects intent, runs hybrid retrieval, and updates long-term profile memory.
    """
    import time
    request_start_time = time.time()
    
    # 1. Fetch or create a session
    if not payload.session_id:
        session = ChatSession(user_id=current_user.id)
        db.add(session)
        db.commit()
        db.refresh(session)
        session_id = session.id
    else:
        session_id = payload.session_id
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
            
    # 2. Fetch recent conversation history (prior to committing the new user message)
    history = await cache.get_session_messages(str(session_id))
    if history is None:
        history = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.desc()).limit(10).all()
        history.reverse() # restore chronological order
        await cache.set_session_messages(str(session_id), history)

    # 3. Save user message to PostgreSQL
    user_msg = ChatMessage(session_id=session_id, role="user", content=payload.message)
    db.add(user_msg)
    db.commit()
    
    # Invalidate session messages cache on new user message
    await cache.invalidate_session(str(session_id))
    
    # 4. Classify query intent using conversation history context
    intent = IntentClassifierService.classify_intent(payload.message, history)
    
    async def sse_event_generator():
        # First event: tell the client the session ID and classified intent
        yield {
            "event": "meta",
            "data": json.dumps({"session_id": str(session_id), "intent": intent})
        }
        
        full_response_content = ""
        
        if intent == "NON_MOVIE":
            # Direct bypass to humorous refusal generator
            for chunk in get_refusal_stream(payload.message):
                full_response_content += chunk
                yield {
                    "event": "content",
                    "data": json.dumps({"text": chunk})
                }
                await asyncio.sleep(0.01) # Small sleep to ensure smooth chunk parsing
        else:
            # Movie query: Fetch profile memory & historical dialogs
            profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
            
            if intent == "MOVIE_RECOMMENDATION":
                # Concierge Discovery Stage: check completeness
                is_eval = (current_user.username == "eval_user_framework")
                is_complete, extracted_profile, follow_up = RecommendationPipeline.assess_completeness(
                    user_id=str(current_user.id),
                    message=payload.message,
                    history=history,
                    db=db,
                    is_evaluation=is_eval
                )
                
                if not is_complete:
                    # Clear out movie cards
                    yield {
                        "event": "movies",
                        "data": json.dumps([])
                    }
                    
                    # Stream follow-up question
                    for chunk in RecommendationPipeline.stream_follow_up_question(follow_up):
                        full_response_content += chunk
                        yield {
                            "event": "content",
                            "data": json.dumps({"text": chunk})
                        }
                        await asyncio.sleep(0.01)
                else:
                    # Profile construction
                    pref_profile = RecommendationPipeline.build_preference_profile(
                        user_id=str(current_user.id),
                        extracted_profile=extracted_profile,
                        db=db
                    )
                    
                    # Caching Layer check
                    cache_key = cache.generate_recommendation_key(pref_profile)
                    cached_rec = await cache.get_recommendation(cache_key)
                    
                    if cached_rec is not None:
                        # Cache Hit: return immediately, bypass search and LLM calls
                        movies_payload = cached_rec["movies"]
                        yield {
                            "event": "movies",
                            "data": json.dumps(movies_payload)
                        }
                        
                        full_response_content = cached_rec["response_text"]
                        
                        # Stream the cached text content in small chunks to preserve the SSE UX
                        chunk_size = 20
                        for i in range(0, len(full_response_content), chunk_size):
                            chunk = full_response_content[i:i+chunk_size]
                            yield {
                                "event": "content",
                                "data": json.dumps({"text": chunk})
                            }
                            await asyncio.sleep(0.01)
                            
                        # Log cached recommendation run
                        total_time = (time.time() - request_start_time) * 1000.0
                        
                        class MockMovie:
                            def __init__(self, id, title):
                                self.id = id
                                self.title = title
                                self.release_date = None
                                self.director = ""
                                self.cast_members = []
                                self.genres = []
                                self.overview = ""
                                self.runtime = 0
                                self.vote_average = 0.0
                                self.popularity = 0.0
                        mock_movies = [MockMovie(m["id"], m["title"]) for m in movies_payload]
                        
                        RecommendationPipeline.log_recommendation(
                            user_id=str(current_user.id),
                            session_id=session_id,
                            profile=pref_profile,
                            query="[CACHED]",
                            retrieved_movies=mock_movies,
                            recommended_movies=mock_movies,
                            response=full_response_content,
                            db=db,
                            original_query=payload.message,
                            cache_hit=True,
                            retrieval_latency_ms=0.0,
                            llm_provider="Cached",
                            llm_latency_ms=0.0,
                            total_response_time_ms=total_time,
                            error_message=None
                        )
                    else:
                        # Cache Miss: run full retrieval and generation
                        error_message = None
                        retrieval_latency = 0.0
                        llm_latency = 0.0
                        retrieved_movies = []
                        retrieval_query = ""
                        
                        try:
                            # 1. Search query generation and hybrid retrieval
                            retrieval_start = time.time()
                            retrieval_query = RecommendationPipeline.generate_retrieval_query(pref_profile)
                            
                            retrieved_movies = RecommendationPipeline.hybrid_retrieve(
                                query=retrieval_query,
                                profile=pref_profile,
                                db=db,
                                client=qdrant,
                                extracted_profile=extracted_profile,
                                original_query=payload.message
                            )
                            retrieval_latency = (time.time() - retrieval_start) * 1000.0
                            
                            # Send movie cards payload
                            movies_payload = [
                                {
                                    "id": m.id,
                                    "title": m.title,
                                    "release_year": m.release_date.year if m.release_date else None,
                                    "director": m.director,
                                    "genres": m.genres,
                                    "vote_average": float(m.vote_average) if m.vote_average else None
                                }
                                for m in retrieved_movies
                            ]
                            yield {
                                "event": "movies",
                                "data": json.dumps(movies_payload)
                            }
                            
                            # 2. Stream recommendations with explanations
                            llm_start = time.time()
                            llm_info = {"provider": "None"}
                            
                            for chunk in RecommendationPipeline.stream_recommendations(
                                movies=retrieved_movies,
                                profile=pref_profile,
                                history=history,
                                current_message=payload.message,
                                llm_info=llm_info
                            ):
                                full_response_content += chunk
                                yield {
                                    "event": "content",
                                    "data": json.dumps({"text": chunk})
                                }
                                await asyncio.sleep(0.01)
                                
                            llm_latency = (time.time() - llm_start) * 1000.0
                            
                            # 3. Store result in Redis cache
                            await cache.set_recommendation(cache_key, {
                                "movies": movies_payload,
                                "response_text": full_response_content
                            })
                            
                            # 4. Persistently log recommendation run
                            total_time = (time.time() - request_start_time) * 1000.0
                            RecommendationPipeline.log_recommendation(
                                user_id=str(current_user.id),
                                session_id=session_id,
                                profile=pref_profile,
                                query=retrieval_query,
                                retrieved_movies=retrieved_movies,
                                recommended_movies=retrieved_movies,
                                response=full_response_content,
                                db=db,
                                original_query=payload.message,
                                cache_hit=False,
                                retrieval_latency_ms=retrieval_latency,
                                llm_provider=llm_info.get("provider", "Gemini"),
                                llm_latency_ms=llm_latency,
                                total_response_time_ms=total_time,
                                error_message=None
                            )
                        except Exception as e:
                            import traceback
                            error_message = traceback.format_exc()
                            total_time = (time.time() - request_start_time) * 1000.0
                            
                            try:
                                RecommendationPipeline.log_recommendation(
                                    user_id=str(current_user.id),
                                    session_id=session_id,
                                    profile=pref_profile,
                                    query=retrieval_query,
                                    retrieved_movies=retrieved_movies,
                                    recommended_movies=[],
                                    response="[ERROR: Recommendation Pipeline Failed]",
                                    db=db,
                                    original_query=payload.message,
                                    cache_hit=False,
                                    retrieval_latency_ms=retrieval_latency,
                                    llm_provider="None",
                                    llm_latency_ms=llm_latency,
                                    total_response_time_ms=total_time,
                                    error_message=error_message
                                )
                            except Exception as db_err:
                                print(f"[Error logging pipeline crash to DB]: {db_err}")
                                
                            yield {
                                "event": "content",
                                "data": json.dumps({"text": "\n[Error] The film projector broke down! Let's restart the scene."})
                            }
                            raise e
            else:
                # Normal movie discussion flow
                movies_payload = []
                retrieved_movies = []
                
                try:
                    discussed_title = IntentClassifierService.extract_discussed_movie(payload.message, history)
                    if discussed_title:
                        from app.models.movie import Movie
                        # Find exact case-insensitive match first
                        movie = db.query(Movie).filter(Movie.title.ilike(discussed_title)).first()
                        if not movie:
                            # Fuzzy matching inside title
                            movie = db.query(Movie).filter(Movie.title.ilike(f"%{discussed_title}%")).order_by(Movie.popularity.desc()).first()
                        
                        if movie:
                            retrieved_movies = [movie]
                            movies_payload = [
                                {
                                    "id": movie.id,
                                    "title": movie.title,
                                    "release_year": movie.release_date.year if movie.release_date else None,
                                    "director": movie.director,
                                    "genres": movie.genres,
                                    "vote_average": float(movie.vote_average) if movie.vote_average else None
                                }
                            ]
                except Exception as ex:
                    print(f"[Warning] Failed to extract/query discussed movie: {ex}")

                yield {
                    "event": "movies",
                    "data": json.dumps(movies_payload)
                }
                
                for chunk in get_chat_stream(
                    message=payload.message,
                    history=history,
                    user_profile=profile,
                    retrieved_movies=retrieved_movies,
                    intent=intent
                ):
                    full_response_content += chunk
                    yield {
                        "event": "content",
                        "data": json.dumps({"text": chunk})
                    }
                    await asyncio.sleep(0.01)
                
        # 4. Save response to PostgreSQL
        # We need a new session in this async generator context to prevent session conflicts
        gen_db = SessionLocal()
        try:
            assistant_msg = ChatMessage(session_id=session_id, role="assistant", content=full_response_content)
            gen_db.add(assistant_msg)
            gen_db.commit()
        finally:
            gen_db.close()
            
        # Invalidate session history cache on new assistant message
        await cache.invalidate_session(str(session_id))
            
        # 5. Consolidate preferences inside the stream if the query hints at a user preference
        pref_keywords = ["like", "love", "dislike", "hate", "prefer", "favorite", "favourite", "director", "actor", "genre", "cinematographer", "writer", "fan of"]
        user_msg_lower = payload.message.lower()
        if any(kw in user_msg_lower for kw in pref_keywords):
            gen_db_mem = SessionLocal()
            try:
                extract_and_consolidate_memory(
                    gen_db_mem,
                    str(current_user.id),
                    payload.message,
                    full_response_content
                )
                yield {
                    "event": "memory_update",
                    "data": json.dumps({"status": "updated"})
                }
            except Exception as e:
                print(f"[Warning] Real-time memory consolidation failed: {e}")
            finally:
                gen_db_mem.close()
        
    return EventSourceResponse(sse_event_generator())

@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a specific chat session and all its message history.
    """
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found."
        )
        
    db.delete(session)
    db.commit()
    return None

@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def submit_recommendation_feedback(
    payload: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submits user feedback (rating and text) for a specific recommendation event.
    """
    from app.models.memory import RecommendationLog, UserFeedback
    
    # Verify the recommendation log exists and belongs to the current user
    log = db.query(RecommendationLog).filter(
        RecommendationLog.id == payload.recommendation_log_id,
        RecommendationLog.user_id == current_user.id
    ).first()
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation log not found."
        )
        
    feedback = UserFeedback(
        recommendation_log_id=payload.recommendation_log_id,
        user_id=current_user.id,
        rating=payload.rating,
        feedback_text=payload.feedback_text
    )
    db.add(feedback)
    db.commit()
    return {"status": "Feedback submitted successfully", "feedback_id": str(feedback.id)}

