import json
import httpx
import google.generativeai as genai
from app.core.config import settings

class IntentClassifierService:
    @classmethod
    def classify_intent(cls, message: str, history: list = None) -> str:
        """
        Classifies the user query intent using Gemini API with optional history context.
        Falls back to Groq API if Gemini fails (e.g. rate limits or server errors).
        Returns one of: 'NON_MOVIE', 'MOVIE_RECOMMENDATION', 'MOVIE_DISCUSSION'
        """
        history_context = ""
        if history:
            history_list = []
            # Utilize the last 3 turns for intent context
            for msg in history[-3:]:
                role = "User" if msg.role == "user" else "CinephileGPT"
                history_list.append(f"{role}: {msg.content}")
            if history_list:
                history_context = "Recent Conversation History:\n" + "\n".join(history_list) + "\n\n"

        prompt = f"""You are the intent classification module for CinephileGPT, an AI system that answers ONLY movie-related questions.
{history_context}Current User Message to classify: "{message}"

Classify the current user message into exactly one of three categories:
1. 'NON_MOVIE': The query is NOT about movies, films, actors, directors, cinema history, film scores, writing screenplays, or film recommendations. Examples: coding advice, math problems, recipe help, fitness routines, weather, general travel guides, or general chat that doesn't reference cinema.
2. 'MOVIE_RECOMMENDATION': The user is explicitly asking for movie recommendations, suggestions, watchlists, or what film they should watch based on criteria (mood, genre, year, similarity to other films), OR they are giving a follow-up instruction to adjust, filter, or correct previous recommendations (e.g., "Yeah but movies from Nolan", "Only horror please", "Suggest something older").
3. 'MOVIE_DISCUSSION': The query is a general statement, question about a movie/director/trivia, or chat that is not asking for new movie recommendations.

Return ONLY a JSON block with the field "intent" representing your classification.

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
