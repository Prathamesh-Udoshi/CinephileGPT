import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.memory import RecommendationLog
from app.models.movie import Movie
from app.core.config import settings

logger = logging.getLogger("analytics_service")

class AnalyticsService:
    @staticmethod
    def get_metrics(db: Session) -> dict[str, Any]:
        """
        Computes various performance and usage metrics from PostgreSQL recommendation logs.
        """
        try:
            # 1. Total and cached recommendation counts
            total_requests = db.query(func.count(RecommendationLog.id)).scalar() or 0
            cache_hits = db.query(func.count(RecommendationLog.id)).filter(RecommendationLog.cache_hit == True).scalar() or 0
            
            # Hit rate calculation
            cache_hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0

            # 2. Average latencies (total, retrieval, LLM) in milliseconds
            avg_total_latency = db.query(func.avg(RecommendationLog.total_response_time_ms)).scalar() or 0.0
            avg_retrieval_latency = db.query(func.avg(RecommendationLog.retrieval_latency_ms)).scalar() or 0.0
            avg_llm_latency = db.query(func.avg(RecommendationLog.llm_latency_ms)).scalar() or 0.0

            # 3. Estimated API Savings (number of calls avoided and cost in dollars)
            cost_per_hit = settings.API_COST_SAVINGS_PER_HIT
            estimated_savings_usd = cache_hits * cost_per_hit

            # 4. Extract most common recommendation intents (genres, moods, themes) from taste profiles
            # Query recent logs to analyze search profiles
            recent_logs = db.query(RecommendationLog.known_information).filter(
                RecommendationLog.known_information.isnot(None)
            ).order_by(RecommendationLog.timestamp.desc()).limit(200).all()

            genres_counts = {}
            moods_counts = {}
            themes_counts = {}

            for (profile,) in recent_logs:
                if not isinstance(profile, dict):
                    continue
                
                # Count Genres
                for genre in profile.get("genres", []):
                    genres_counts[genre] = genres_counts.get(genre, 0) + 1
                
                # Count Moods
                for mood in profile.get("moods", []):
                    moods_counts[mood] = moods_counts.get(mood, 0) + 1
                
                # Count Themes
                for theme in profile.get("themes", []):
                    themes_counts[theme] = themes_counts.get(theme, 0) + 1

            # Sort and pick top 5
            top_genres = sorted([{"genre": k, "count": v} for k, v in genres_counts.items()], key=lambda x: x["count"], reverse=True)[:5]
            top_moods = sorted([{"mood": k, "count": v} for k, v in moods_counts.items()], key=lambda x: x["count"], reverse=True)[:5]
            top_themes = sorted([{"theme": k, "count": v} for k, v in themes_counts.items()], key=lambda x: x["count"], reverse=True)[:5]

            # 5. Determine recommendation popularity (most frequently recommended movies)
            all_rec_movie_ids = db.query(RecommendationLog.recommended_movie_ids).filter(
                RecommendationLog.recommended_movie_ids.isnot(None)
            ).all()

            movie_frequencies = {}
            for (movie_ids,) in all_rec_movie_ids:
                if not movie_ids:
                    continue
                for m_id in movie_ids:
                    movie_frequencies[m_id] = movie_frequencies.get(m_id, 0) + 1

            # Get top 5 popular movie IDs
            top_popular_ids = sorted(movie_frequencies.keys(), key=lambda x: movie_frequencies[x], reverse=True)[:5]
            
            top_movies = []
            if top_popular_ids:
                movies = db.query(Movie).filter(Movie.id.in_(top_popular_ids)).all()
                movies_map = {m.id: m for m in movies}
                for m_id in top_popular_ids:
                    if m_id in movies_map:
                        top_movies.append({
                            "movie_id": m_id,
                            "title": movies_map[m_id].title,
                            "director": movies_map[m_id].director,
                            "count": movie_frequencies[m_id]
                        })

            return {
                "performance": {
                    "total_requests": total_requests,
                    "cache_hits": cache_hits,
                    "cache_hit_rate": round(cache_hit_rate, 4),
                    "avg_total_latency_ms": round(avg_total_latency, 2),
                    "avg_retrieval_latency_ms": round(avg_retrieval_latency, 2),
                    "avg_llm_latency_ms": round(avg_llm_latency, 2)
                },
                "savings": {
                    "api_calls_saved": cache_hits,
                    "estimated_savings_usd": round(estimated_savings_usd, 4),
                    "cost_per_hit_usd": cost_per_hit
                },
                "top_intents": {
                    "genres": top_genres,
                    "moods": top_moods,
                    "themes": top_themes
                },
                "popular_recommendations": top_movies
            }

        except Exception as e:
            logger.error(f"Error computing analytics: {e}")
            return {
                "error": f"Failed to compute analytics: {str(e)}",
                "performance": {},
                "savings": {},
                "top_intents": {},
                "popular_recommendations": []
            }
