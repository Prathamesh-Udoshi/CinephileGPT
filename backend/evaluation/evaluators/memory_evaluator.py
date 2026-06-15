from evaluation.evaluators.base_evaluator import BaseEvaluator

class MemoryEvaluator(BaseEvaluator):
    def evaluate_case(self, case: dict) -> dict:
        case_id = case.get("id")
        query = case.get("query")
        memory_profile = case.get("memory", {})
        conversation_history = case.get("conversation_history", [])
        expected_behavior = ", ".join(case.get("expected_behavior", []))
        avoid = ", ".join(case.get("avoid", []))

        # Setup state
        session = self._reset_eval_context(
            profile_data=memory_profile,
            conversation_history=conversation_history
        )
        
        # Run assistant
        response = self._execute_assistant(query, session)
        
        # Judge
        judge_result = self.judge.evaluate(
            category="Memory Evaluation",
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
