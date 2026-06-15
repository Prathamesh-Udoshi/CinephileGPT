import argparse
import json
import os
import sys
import time
import traceback
# Add the backend directory to sys.path so app and evaluation imports work
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from datetime import datetime
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, get_qdrant, Base, engine
from evaluation.models import EvaluationRun, EvaluationCaseResult
from evaluation.judges.llm_judge import LLMJudge

# Evaluators
from evaluation.evaluators.recommendation_evaluator import RecommendationEvaluator
from evaluation.evaluators.personality_evaluator import PersonalityEvaluator
from evaluation.evaluators.memory_evaluator import MemoryEvaluator
from evaluation.evaluators.refusal_evaluator import RefusalEvaluator
from evaluation.evaluators.retrieval_evaluator import RetrievalEvaluator

DATASET_FILES = {
    "recommendation": "recommendation_eval.json",
    "personality": "personality_eval.json",
    "memory": "memory_eval.json",
    "retrieval": "retrieval_eval.json",
    "refusal": "refusal_eval.json"
}

def load_dataset(category: str) -> list:
    """
    Loads the JSON dataset for a specific category.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "datasets", DATASET_FILES[category])
    
    if not os.path.exists(file_path):
        # Fallback to evaluation directory (if not moved yet)
        file_path = os.path.join(base_dir, DATASET_FILES[category])
        if not os.path.exists(file_path):
            print(f"[Error] Dataset file not found for category {category}")
            return []
            
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation():
    parser = argparse.ArgumentParser(description="CinephileGPT AI Evaluation Framework Runner")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Limit the number of test cases run per category (for fast iteration)")
    parser.add_argument("-c", "--category", type=str, choices=list(DATASET_FILES.keys()), default=None,
                        help="Run evaluation for a specific category only")
    parser.add_argument("-i", "--case-id", type=str, default=None,
                        help="Run a specific test case by ID (e.g. rec_001)")
    parser.add_argument("--save-db", type=str, choices=["true", "false"], default="true",
                        help="Save results and run details to PostgreSQL history")
    args = parser.parse_args()

    save_to_db = args.save_db == "true"

    print("=" * 60)
    print("=== CinephileGPT Production Evaluation Framework ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Initialize DB tables if they don't exist
    try:
        print("[DB] Initializing PostgreSQL evaluation schema...")
        Base.metadata.create_all(bind=engine)
        print("[DB] Schema initialized successfully.")
    except Exception as e:
        print(f"[DB Error] Failed to connect/initialize PostgreSQL: {e}")
        print("Please check your .env settings.")
        sys.exit(1)

    db: Session = SessionLocal()
    qdrant = get_qdrant()
    judge = LLMJudge()

    # 2. Select categories and load test cases
    categories_to_run = [args.category] if args.category else list(DATASET_FILES.keys())
    
    all_test_cases = {}
    for cat in categories_to_run:
        cases = load_dataset(cat)
        # Apply filters
        if args.case_id:
            cases = [c for c in cases if c.get("id") == args.case_id]
        elif args.limit:
            cases = cases[:args.limit]
        
        if cases:
            all_test_cases[cat] = cases

    total_loaded = sum(len(c) for c in all_test_cases.values())
    if total_loaded == 0:
        print("[Error] No test cases loaded. Exiting.")
        db.close()
        return

    print(f"[Info] Loaded {total_loaded} test cases across {len(all_test_cases)} categories.")
    for cat, cases in all_test_cases.items():
        print(f"  - {cat.capitalize()}: {len(cases)} cases")

    # 3. Create database Run record
    run_record = None
    if save_to_db:
        run_record = EvaluationRun(
            status="running",
            total_cases=total_loaded
        )
        db.add(run_record)
        db.commit()
        db.refresh(run_record)
        print(f"[DB] Registered Run #{run_record.id} in history database.")

    # 4. Initialize Evaluators mapping
    evaluators_map = {
        "recommendation": RecommendationEvaluator(db, qdrant, judge),
        "personality": PersonalityEvaluator(db, qdrant, judge),
        "memory": MemoryEvaluator(db, qdrant, judge),
        "refusal": RefusalEvaluator(db, qdrant, judge),
        "retrieval": RetrievalEvaluator(db, qdrant, judge)
    }

    # 5. Run test cases
    run_results = []
    category_aggregates = {}

    try:
        case_index = 0
        for cat, cases in all_test_cases.items():
            print("\n" + "-" * 50)
            print(f"[Category] Evaluating: {cat.upper()}")
            print("-" * 50)
            
            evaluator = evaluators_map[cat]
            cat_passed = 0
            cat_scores = []

            for case in cases:
                case_index += 1
                case_id = case.get("id")
                difficulty = case.get("difficulty", "N/A")
                print(f"[{case_index}/{total_loaded}] Running Case {case_id} (Diff: {difficulty}) - Query: '{case.get('query')[:40]}...'")
                
                # Execution
                start_time = time.time()
                try:
                    result = evaluator.evaluate_case(case)
                    elapsed = time.time() - start_time
                    
                    passed = result["passed"]
                    score = result["score"]
                    cat_scores.append(score)
                    if passed:
                        cat_passed += 1

                    print(f"      Result: {'PASSED' if passed else 'FAILED'} | Score: {score:.1f}/10.0 | Time: {elapsed:.2f}s")
                    
                    # Store run details
                    case_result = {
                        "case_id": case_id,
                        "category": cat,
                        "difficulty": difficulty,
                        "query": case.get("query"),
                        "expected": case,
                        "actual_response": result["actual_response"],
                        "passed": passed,
                        "score": score,
                        "sub_scores": result["sub_scores"],
                        "strengths": result["strengths"],
                        "weaknesses": result["weaknesses"],
                        "reasoning": result["reasoning"]
                    }
                    run_results.append(case_result)

                    # Save case result in Postgres
                    if save_to_db and run_record:
                        db_case = EvaluationCaseResult(
                            run_id=run_record.id,
                            case_id=case_id,
                            category=cat,
                            difficulty=difficulty if difficulty != "N/A" else None,
                            query=case.get("query"),
                            expected=case,
                            actual_response=result["actual_response"],
                            passed=passed,
                            score=score,
                            sub_scores=result["sub_scores"],
                            strengths=result["strengths"],
                            weaknesses=result["weaknesses"],
                            reasoning=result["reasoning"]
                        )
                        db.add(db_case)
                        db.commit()

                except Exception as eval_err:
                    print(f"      [Error] executing case {case_id}: {eval_err}")
                    traceback.print_exc()
                    
                # Small rate-limit protection sleep
                time.sleep(0.5)

            # Record Category Averages
            if cat_scores:
                avg_score = sum(cat_scores) / len(cat_scores)
                pass_rate = cat_passed / len(cat_scores)
            else:
                avg_score = 0.0
                pass_rate = 0.0
                
            category_aggregates[cat] = {
                "avg_score": round(avg_score, 2),
                "pass_rate": round(pass_rate, 2),
                "total": len(cases),
                "passed": cat_passed,
                "failed": len(cases) - cat_passed
            }
            print(f"[Summary] Category {cat.capitalize()}: Avg Score = {avg_score:.2f}/10.0 | Pass Rate = {pass_rate*100:.1f}%")

        # 6. Calculate Overall Run Summary
        overall_scores = [r["score"] for r in run_results]
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        passed_cases = sum(1 for r in run_results if r["passed"])
        pass_rate = passed_cases / len(run_results) if run_results else 0.0

        print("\n" + "=" * 60)
        print("=== EVALUATION RUN COMPLETED SUMMARY ===")
        print(f"Overall System Score: {overall_score:.2f}/10.0")
        print(f"Pass Rate: {pass_rate*100:.1f}% ({passed_cases}/{len(run_results)} cases)")
        print("=" * 60)

        # Find best and worst category
        best_cat = max(category_aggregates.keys(), key=lambda k: category_aggregates[k]["avg_score"])
        worst_cat = min(category_aggregates.keys(), key=lambda k: category_aggregates[k]["avg_score"])
        print(f"Best Performing Category: {best_cat.capitalize()} ({category_aggregates[best_cat]['avg_score']}/10.0)")
        print(f"Worst Performing Category: {worst_cat.capitalize()} ({category_aggregates[worst_cat]['avg_score']}/10.0)")

        # Write reports files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(base_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # JSON results dump
        results_path = os.path.join(reports_dir, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as rf:
            json.dump(run_results, rf, indent=2)

        # JSON summary dump
        summary_payload = {
            "run_id": run_record.id if run_record else None,
            "timestamp": datetime.now().isoformat(),
            "overall_score": round(overall_score, 2),
            "pass_rate": round(pass_rate, 4),
            "total_cases": len(run_results),
            "passed_cases": passed_cases,
            "failed_cases": len(run_results) - passed_cases,
            "best_category": best_cat,
            "worst_category": worst_cat,
            "category_summary": category_aggregates
        }
        summary_path = os.path.join(reports_dir, "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary_payload, sf, indent=2)

        # JSON category reports
        cat_reports_path = os.path.join(reports_dir, "category_reports.json")
        with open(cat_reports_path, "w", encoding="utf-8") as cf:
            json.dump(category_aggregates, cf, indent=2)

        print(f"[System] Exported structured reports successfully to {reports_dir}/")

        # 7. Update database record with final scores
        if save_to_db and run_record:
            run_record.overall_score = round(overall_score, 2)
            run_record.pass_rate = round(pass_rate, 4)
            run_record.passed_cases = passed_cases
            run_record.failed_cases = len(run_results) - passed_cases
            
            # Scores
            run_record.recommendation_score = category_aggregates.get("recommendation", {}).get("avg_score")
            run_record.personality_score = category_aggregates.get("personality", {}).get("avg_score")
            run_record.memory_score = category_aggregates.get("memory", {}).get("avg_score")
            run_record.retrieval_score = category_aggregates.get("retrieval", {}).get("avg_score")
            run_record.refusal_score = category_aggregates.get("refusal", {}).get("avg_score")

            # Pass Rates
            run_record.recommendation_pass_rate = category_aggregates.get("recommendation", {}).get("pass_rate")
            run_record.personality_pass_rate = category_aggregates.get("personality", {}).get("pass_rate")
            run_record.memory_pass_rate = category_aggregates.get("memory", {}).get("pass_rate")
            run_record.retrieval_pass_rate = category_aggregates.get("retrieval", {}).get("pass_rate")
            run_record.refusal_pass_rate = category_aggregates.get("refusal", {}).get("pass_rate")

            run_record.status = "completed"
            db.commit()
            print(f"[DB] Evaluation run Run #{run_record.id} updated to 'completed'.")

    except Exception as run_err:
        print(f"[Fatal Run Error] {run_err}")
        traceback.print_exc()
        if save_to_db and run_record:
            run_record.status = "failed"
            db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    run_evaluation()
