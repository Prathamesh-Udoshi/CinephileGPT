import json
import httpx
import google.generativeai as genai
from app.core.config import settings

class IntentClassifierService:
    @classmethod
    def classify_intent(cls, message: str) -> str:
        """
        Classifies the user query intent using Gemini API.
        Falls back to Groq API if Gemini fails (e.g. rate limits or server errors).
        Returns one of: 'NON_MOVIE', 'MOVIE_RECOMMENDATION', 'MOVIE_DISCUSSION'
        """
        prompt = f"""You are the intent classification module for CinephileGPT, an AI system that answers ONLY movie-related questions.
Classify the following user message into exactly one of three categories:
1. 'NON_MOVIE': The query is NOT about movies, films, actors, directors, cinema history, film scores, writing screenplays, or film recommendations. Examples: coding advice, math problems, recipe help, fitness routines, weather, general travel guides, or general chat that doesn't reference cinema.
2. 'MOVIE_RECOMMENDATION': The user is explicitly asking for movie recommendations, suggestions, watchlists, or what film they should watch based on criteria (mood, genre, year, similarity to other films).
3. 'MOVIE_DISCUSSION': The query is about movies, actors, directors, film history, reviews, explanation of a film's ending, film techniques (cinematography, lighting), or cinematic trivia.

Return ONLY a JSON block with the field "intent" representing your classification.

User Message: "{message}"

JSON Response:"""

        # 1. Try Gemini first (if key configured)
        if settings.GEMINI_API_KEY:
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
                data = json.loads(text.strip())
                intent = data.get("intent", "").upper()
                if intent in ["NON_MOVIE", "MOVIE_RECOMMENDATION", "MOVIE_DISCUSSION"]:
                    return intent
            except Exception as e:
                print(f"[Warning] Gemini intent classification failed: {e}. Attempting Groq fallback...")

        # 2. Try Groq as fallback
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
                r = httpx.post(url, headers=headers, json=payload, timeout=10.0)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    resp_json = json.loads(content.strip())
                    intent = resp_json.get("intent", "").upper()
                    if intent in ["NON_MOVIE", "MOVIE_RECOMMENDATION", "MOVIE_DISCUSSION"]:
                        return intent
                else:
                    print(f"[Warning] Groq intent API returned {r.status_code}: {r.text}")
            except Exception as e:
                print(f"[Error] Groq intent classification failed: {e}")

        # Default fallback
        return "MOVIE_DISCUSSION"
