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
from app.schemas.chat import ChatSessionResponse, ChatStreamRequest
from app.services.intent import IntentClassifierService
from app.services.retrieval import hybrid_retrieval
from app.services.llm import get_refusal_stream, get_chat_stream, extract_and_consolidate_memory

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
def stream_chat(
    payload: ChatStreamRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    qdrant = Depends(get_qdrant)
):
    """
    Core conversational streaming route using Server-Sent Events (SSE).
    Detects intent, runs hybrid retrieval, and updates long-term profile memory.
    """
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
            
    # 2. Save user message to PostgreSQL
    user_msg = ChatMessage(session_id=session_id, role="user", content=payload.message)
    db.add(user_msg)
    db.commit()
    
    # 3. Classify query intent
    intent = IntentClassifierService.classify_intent(payload.message)
    
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
            history = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.timestamp.desc()).offset(1).limit(10).all()
            history.reverse() # restore chronological order
            
            # Fetch retrieved context ONLY if the user is explicitly asking for recommendations
            if intent == "MOVIE_RECOMMENDATION":
                retrieved_movies = hybrid_retrieval(
                    db=db,
                    client=qdrant,
                    query=payload.message,
                    limit=5,
                    user_profile={
                        "favorite_genres": profile.favorite_genres if profile else [],
                        "disliked_genres": profile.disliked_genres if profile else []
                    } if profile else None
                )
            else:
                retrieved_movies = []
            
            # Send movie cards payload to frontend
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
            
            # Trigger customized persona response stream
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
