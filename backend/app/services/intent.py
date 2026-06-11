import json
import google.generativeai as genai
from app.core.config import settings

class IntentClassifierService:
    @classmethod
    def classify_intent(cls, message: str) -> str:
        """
        Classifies the user query intent using Gemini API.
        Returns one of: 'NON_MOVIE', 'MOVIE_RECOMMENDATION', 'MOVIE_DISCUSSION'
        """
        # If the API key is not set, default to MOVIE_DISCUSSION to allow normal operations
        if not settings.GEMINI_API_KEY:
            return "MOVIE_DISCUSSION"

        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
            
            prompt = f"""You are the intent classification module for CinephileGPT, an AI system that answers ONLY movie-related questions.
Classify the following user message into exactly one of three categories:
1. 'NON_MOVIE': The query is NOT about movies, films, actors, directors, cinema history, film scores, writing screenplays, or film recommendations. Examples: coding advice, math problems, recipe help, fitness routines, weather, general travel guides, or general chat that doesn't reference cinema.
2. 'MOVIE_RECOMMENDATION': The user is explicitly asking for movie recommendations, suggestions, watchlists, or what film they should watch based on criteria (mood, genre, year, similarity to other films).
3. 'MOVIE_DISCUSSION': The query is about movies, actors, directors, film history, reviews, explanation of a film's ending, film techniques (cinematography, lighting), or cinematic trivia.

Return ONLY a JSON block with the field "intent" representing your classification.

User Message: "{message}"

JSON Response:"""

            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            data = json.loads(response.text.strip())
            intent = data.get("intent", "MOVIE_DISCUSSION").upper()
            
            if intent in ["NON_MOVIE", "MOVIE_RECOMMENDATION", "MOVIE_DISCUSSION"]:
                return intent
            return "MOVIE_DISCUSSION"
            
        except Exception as e:
            # Robust error boundary: log the error and default to movie discussion
            print(f"Error classifying intent: {e}")
            return "MOVIE_DISCUSSION"
