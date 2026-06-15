import json
import re
import google.generativeai as genai
from app.core.config import settings
import httpx

JUDGE_PROMPT_TEMPLATE = """You are an expert AI Evaluation Judge.

Your task is to evaluate the quality of an AI assistant response.

You must evaluate objectively and consistently.

Return ONLY valid JSON.

Evaluation Category:
{category}

User Query:
{query}

Expected Behavior:
{expected_behavior}

Things To Avoid:
{avoid}

Assistant Response:
{response}

Scoring Rules:

For recommendation evaluation:

* Relevance: 1-10
* Personalization: 1-10
* Explanation Quality: 1-10
* Diversity: 1-10
* Overall Score: 1-10

For personality evaluation:

* Personality Consistency: 1-10
* Movie Enthusiasm: 1-10
* Humor: 1-10
* Character Preservation: 1-10
* Overall Score: 1-10

For memory evaluation:

* Memory Recall: 1-10
* Preference Usage: 1-10
* Continuity: 1-10
* Overall Score: 1-10

For refusal evaluation:

* Refusal Compliance: 1-10
* Character Preservation: 1-10
* Redirect Quality: 1-10
* Overall Score: 1-10

Provide:

{{
"passed": true,
"overall_score": 8.5,
"sub_scores": {{}},
"strengths": [],
"weaknesses": [],
"reasoning": ""
}}

Be strict.
Do not inflate scores.
Penalize generic responses.
Penalize hallucinations.
Penalize ignoring memory.
Penalize breaking character.
Penalize weak refusals.
Reward helpfulness, personality consistency, and correct behavior."""

class LLMJudge:
    def __init__(self):
        # Configure APIs
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
        
    def _clean_and_parse_json(self, text: str) -> dict:
        """
        Extract and parse a JSON block from the LLM response.
        Uses regex and strip cleaning for robustness.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try regex match for code blocks
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # General cleanup fallback
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Last resort: try to extract anything that looks like an object
                obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
                if obj_match:
                    try:
                        return json.loads(obj_match.group(1))
                    except json.JSONDecodeError:
                        pass
                return None

    def evaluate(self, category: str, query: str, expected_behavior: str, avoid: str, response: str) -> dict:
        """
        Evaluates a single test case response using LLM-as-a-Judge.
        Gemini is the primary evaluator; Groq is the fallback.
        """
        # Format the prompt
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            category=category,
            query=query,
            expected_behavior=expected_behavior,
            avoid=avoid,
            response=response
        )

        judge_response_text = None

        # 1. Try Gemini
        if settings.GEMINI_API_KEY:
            try:
                # Use Gemini 2.5 flash or settings.GEMINI_MODEL_NAME
                model = genai.GenerativeModel(
                    settings.GEMINI_MODEL_NAME
                )
                res = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                judge_response_text = res.text
            except Exception as e:
                print(f"[Judge Warning] Gemini evaluation failed: {e}. Falling back to Groq...")

        # 2. Try Groq Fallback
        if not judge_response_text and settings.GROQ_API_KEY:
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
                    "temperature": 0.1
                }
                r = httpx.post(url, headers=headers, json=payload, timeout=20.0)
                if r.status_code == 200:
                    data = r.json()
                    judge_response_text = data["choices"][0]["message"]["content"]
                else:
                    print(f"[Judge Error] Groq evaluation API returned {r.status_code}: {r.text}")
            except Exception as e:
                print(f"[Judge Error] Groq evaluation fallback failed: {e}")

        # Parse output
        if judge_response_text:
            parsed = self._clean_and_parse_json(judge_response_text)
            if parsed:
                # Sanitize response fields to ensure keys are present
                sanitized = {
                    "passed": parsed.get("passed", False),
                    "overall_score": float(parsed.get("overall_score", 0.0)),
                    "sub_scores": parsed.get("sub_scores", {}),
                    "strengths": parsed.get("strengths", []),
                    "weaknesses": parsed.get("weaknesses", []),
                    "reasoning": parsed.get("reasoning", "No reasoning provided by Judge.")
                }
                return sanitized

        # Complete Failure Fallback
        return {
            "passed": False,
            "overall_score": 0.0,
            "sub_scores": {},
            "strengths": [],
            "weaknesses": ["Judge API timeout/error"],
            "reasoning": "Both Gemini and Groq API backends encountered errors or rate limits."
        }
