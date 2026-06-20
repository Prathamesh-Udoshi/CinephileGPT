import abc
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from app.models.user import User
from app.models.memory import UserProfile, ChatSession, ChatMessage, RecommendationLog
from app.services.intent import IntentClassifierService
from app.services.retrieval import hybrid_retrieval
from app.services.llm import get_refusal_stream, get_chat_stream
from app.services.recommendation import RecommendationPipeline
from evaluation.judges.llm_judge import LLMJudge

class BaseEvaluator(abc.ABC):
    def __init__(self, db: Session, qdrant: QdrantClient, judge: LLMJudge):
        self.db = db
        self.qdrant = qdrant
        self.judge = judge
        self.eval_user = self._get_or_create_eval_user()

    def _get_or_create_eval_user(self) -> User:
        """
        Retrieves or creates a dedicated evaluation user to prevent state pollution in production.
        """
        user = self.db.query(User).filter(User.username == "eval_user_framework").first()
        if not user:
            user = User(
                username="eval_user_framework",
                email="eval_user_framework@example.com",
                password_hash="eval_dummy_password_hash_12345"
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            profile = UserProfile(
                user_id=user.id,
                favorite_genres=[],
                favorite_directors=[],
                favorite_actors=[],
                disliked_genres=[],
                general_notes=""
            )
            self.db.add(profile)
            self.db.commit()
            
        return user

    def _reset_eval_context(self, profile_data: dict = None, conversation_history: list = None) -> ChatSession:
        """
        Clears out old profile memory and message history, then populates fresh preferences 
        and chat history for a specific test case execution.
        """
        # 1. Update Profile
        profile = self.db.query(UserProfile).filter(UserProfile.user_id == self.eval_user.id).first()
        if not profile:
            profile = UserProfile(user_id=self.eval_user.id)
            self.db.add(profile)

        profile_data = profile_data or {}
        
        # Pull standard lists
        profile.favorite_genres = profile_data.get("favorite_genres", [])
        if "favorite_genre" in profile_data and profile_data["favorite_genre"]:
            profile.favorite_genres = list(set(profile.favorite_genres + [profile_data["favorite_genre"]]))
            
        profile.favorite_directors = profile_data.get("favorite_directors", [])
        if "favorite_director" in profile_data and profile_data["favorite_director"]:
            profile.favorite_directors = list(set(profile.favorite_directors + [profile_data["favorite_director"]]))
            
        profile.favorite_actors = profile_data.get("favorite_actors", [])
        if "favorite_actor" in profile_data and profile_data["favorite_actor"]:
            profile.favorite_actors = list(set(profile.favorite_actors + [profile_data["favorite_actor"]]))
            
        profile.disliked_genres = profile_data.get("disliked_genres", [])
        if "disliked_genre" in profile_data and profile_data["disliked_genre"]:
            profile.disliked_genres = list(set(profile.disliked_genres + [profile_data["disliked_genre"]]))

        # Serialize any other custom fields as general notes for LLM context inclusion
        special_notes = []
        for k, v in profile_data.items():
            if k not in ["favorite_genres", "favorite_genre", "favorite_directors", "favorite_director", 
                         "favorite_actors", "favorite_actor", "disliked_genres", "disliked_genre"]:
                special_notes.append(f"{k}: {v}")
                
        profile.general_notes = " | ".join(special_notes) if special_notes else ""
        self.db.commit()

        # 2. Reset Chat History
        # Delete old sessions
        self.db.query(ChatSession).filter(ChatSession.user_id == self.eval_user.id).delete()
        self.db.query(RecommendationLog).filter(RecommendationLog.user_id == self.eval_user.id).delete()
        self.db.commit()

        # Create new session
        session = ChatSession(user_id=self.eval_user.id)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

        # Populate history
        if conversation_history:
            for text_line in conversation_history:
                # Expecting format 'User: ...' or 'Assistant: ...' or general text lines
                text_line_str = str(text_line)
                if text_line_str.lower().startswith("user:"):
                    role = "user"
                    content = text_line_str[5:].strip()
                elif text_line_str.lower().startswith("assistant:"):
                    role = "assistant"
                    content = text_line_str[10:].strip()
                else:
                    role = "user"
                    content = text_line_str.strip()

                msg = ChatMessage(session_id=session.id, role=role, content=content)
                self.db.add(msg)
            self.db.commit()

        return session

    def _execute_assistant(self, query: str, session: ChatSession) -> str:
        """
        Executes CinephileGPT response generation pipeline programmatically.
        Matches the intent classification, retrieval, and LLM orchestration of main.py.
        """
        profile = self.db.query(UserProfile).filter(UserProfile.user_id == self.eval_user.id).first()
        history = self.db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.timestamp.desc()).limit(10).all()
        history.reverse()

        intent = IntentClassifierService.classify_intent(query)
        
        response_text = ""
        if intent == "NON_MOVIE":
            for chunk in get_refusal_stream(query):
                response_text += chunk
        elif intent == "MOVIE_RECOMMENDATION":
            # Set is_evaluation=True to bypass clarification questions during evaluation runs
            is_complete, extracted_profile, follow_up = RecommendationPipeline.assess_completeness(
                user_id=str(self.eval_user.id),
                message=query,
                history=history,
                db=self.db,
                is_evaluation=True
            )
            
            if not is_complete:
                for chunk in RecommendationPipeline.stream_follow_up_question(follow_up):
                    response_text += chunk
            else:
                pref_profile = RecommendationPipeline.build_preference_profile(
                    user_id=str(self.eval_user.id),
                    extracted_profile=extracted_profile,
                    db=self.db
                )
                
                retrieval_query = RecommendationPipeline.generate_retrieval_query(pref_profile)
                
                retrieved_movies = RecommendationPipeline.hybrid_retrieve(
                    query=retrieval_query,
                    profile=pref_profile,
                    db=self.db,
                    client=self.qdrant,
                    extracted_profile=extracted_profile,
                    original_query=query
                )
                
                for chunk in RecommendationPipeline.stream_recommendations(
                    movies=retrieved_movies,
                    profile=pref_profile,
                    history=history,
                    current_message=query
                ):
                    response_text += chunk
                    
                RecommendationPipeline.log_recommendation(
                    user_id=str(self.eval_user.id),
                    session_id=session.id,
                    profile=pref_profile,
                    query=retrieval_query,
                    retrieved_movies=retrieved_movies,
                    recommended_movies=retrieved_movies,
                    response=response_text,
                    db=self.db
                )
        else: # MOVIE_DISCUSSION
            for chunk in get_chat_stream(
                message=query,
                history=history,
                user_profile=profile,
                retrieved_movies=[],
                intent=intent
            ):
                response_text += chunk

        return response_text.strip()

    @abc.abstractmethod
    def evaluate_case(self, case: dict) -> dict:
        """
        Runs evaluation for a single case. Returns a dict in format:
        {
           "case_id": str,
           "passed": bool,
           "score": float,
           "actual_response": str,
           "sub_scores": dict,
           "strengths": list,
           "weaknesses": list,
           "reasoning": str
        }
        """
        pass
