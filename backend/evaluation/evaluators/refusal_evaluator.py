from evaluation.evaluators.base_evaluator import BaseEvaluator

class RefusalEvaluator(BaseEvaluator):
    def evaluate_case(self, case: dict) -> dict:
        case_id = case.get("id")
        query = case.get("query")
        expected_behavior = ", ".join(case.get("expected_behavior", []))
        avoid = ", ".join(case.get("avoid", []))

        # Setup state
        session = self._reset_eval_context()
        
        # Run assistant
        response = self._execute_assistant(query, session)
        
        # Judge
        judge_result = self.judge.evaluate(
            category="Refusal Evaluation",
            query=query,
            expected_behavior=expected_behavior,
            avoid=avoid,
            response=response
        )

        return {
            "case_id": case_id,
            "passed": judge_result["passed"],
            "score": judge_result["overall_score"],
            "actual_response": response,
            "sub_scores": judge_result["sub_scores"],
            "strengths": judge_result["strengths"],
            "weaknesses": judge_result["weaknesses"],
            "reasoning": judge_result["reasoning"]
        }
