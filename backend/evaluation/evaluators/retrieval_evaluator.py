from evaluation.evaluators.base_evaluator import BaseEvaluator
from evaluation.metrics.retrieval_metrics import (
    compute_recall_at_k,
    compute_hit_rate,
    compute_reciprocal_rank,
    compute_retrieval_accuracy
)
from app.services.retrieval import hybrid_retrieval

class RetrievalEvaluator(BaseEvaluator):
    def evaluate_case(self, case: dict) -> dict:
        case_id = case.get("id")
        query = case.get("query")
        expected_movies = case.get("expected_movies", [])
        must_not_retrieve = case.get("must_not_retrieve", [])

        # Execute hybrid retrieval directly
        # We retrieve up to 10 movies so we can compute metrics up to rank 10
        retrieved_movies = hybrid_retrieval(
            db=self.db,
            client=self.qdrant,
            query=query,
            limit=10,
            user_profile=None
        )
        
        retrieved_titles = [m.title for m in retrieved_movies]

        # Calculate programmatic metrics
        r5 = compute_recall_at_k(retrieved_titles, expected_movies, 5)
        r10 = compute_recall_at_k(retrieved_titles, expected_movies, 10)
        hit_rate = compute_hit_rate(retrieved_titles, expected_movies, 10)
        mrr = compute_reciprocal_rank(retrieved_titles, expected_movies, 10)
        accuracy = compute_retrieval_accuracy(retrieved_titles, expected_movies, must_not_retrieve, 10)

        # We scale metrics from [0.0, 1.0] to [1.0, 10.0] for dashboard scorecard consistency
        sub_scores = {
            "Recall@5": round(r5 * 10.0, 2),
            "Recall@10": round(r10 * 10.0, 2),
            "Hit Rate": round(hit_rate * 10.0, 2),
            "MRR": round(mrr * 10.0, 2),
            "Retrieval Accuracy": round(accuracy * 10.0, 2)
        }

        # A case is considered passed if retrieval accuracy is at least 70%
        passed = accuracy >= 0.70

        # Strengths & Weaknesses
        strengths = []
        weaknesses = []
        
        retrieved_norm = {t.lower().strip() for t in retrieved_titles}
        for exp in expected_movies:
            if exp.lower().strip() in retrieved_norm:
                strengths.append(f"Successfully retrieved expected movie: {exp}")
            else:
                weaknesses.append(f"Missed expected movie: {exp}")

        for mn in must_not_retrieve:
            if mn.lower().strip() in retrieved_norm:
                weaknesses.append(f"Violated exclusion rule by retrieving: {mn}")
                passed = False # Force fail if forbidden item retrieved

        actual_str = ", ".join(retrieved_titles) if retrieved_titles else "No movies retrieved."
        expected_str = ", ".join(expected_movies)
        must_not_str = ", ".join(must_not_retrieve) if must_not_retrieve else "None"
        
        reasoning = (
            f"Retrieved: [{actual_str}].\n"
            f"Expected: [{expected_str}].\n"
            f"Exclusions: [{must_not_str}].\n"
            f"Recall@5: {r5:.2f}, Recall@10: {r10:.2f}, Hit Rate: {hit_rate:.2f}, MRR: {mrr:.2f}, Accuracy: {accuracy:.2f}."
        )

        return {
            "case_id": case_id,
            "passed": passed,
            "score": round(accuracy * 10.0, 2),
            "actual_response": f"Retrieved Movies: {retrieved_titles}",
            "sub_scores": sub_scores,
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "reasoning": reasoning
        }
